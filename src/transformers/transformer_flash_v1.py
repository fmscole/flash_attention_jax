"""
Transformer 基础模块 — 基于 Flash Attention (stax_plus.flash_attention)

提供可复用的 Transformer 构建块，所有层遵循 stax (init_fun, apply_fun) 约定，
可被其他模块（如 ViT、翻译模型等）直接导入使用。

导出接口:
    MultiHeadSelfAttention   — 多头自注意力（FlashAttention）
    MultiHeadCrossAttention  — 多头交叉注意力（FlashAttention）
    TransformerEncoderBlock  — Pre-LN 编码器块
    TransformerDecoderBlock  — Pre-LN 解码器块
    TransformerEncoder       — 编码器堆栈
    TransformerDecoder       — 解码器堆栈
    Transformer              — 完整 Encoder-Decoder Transformer

【数值稳定性说明 — 2024 修复】
flash_attention 在分块模式 (block_size < seq_len) 下存在过一个 epsilon 不一致 bug：
前向归一化和 log_sum_exp 保存使用了不同的 epsilon，导致反向传播恢复的 attention
权重与前向不一致。该 bug 已修复（统一使用 _MIN_NORMALIZER=1e-12），
分块模式现已数值精确一致。本模块所有注意力层均通过 flash_attention 间接获益。
"""

import jax
import jax.numpy as jnp
from jax import lax, random
from functools import partial

from lib import stax_plus as stax
from lib.stax_plus import (
    Dense, Dropout, Gelu, LayerNorm,
    FanOut, FanInSum, FanInConcat, Identity, serial, parallel,
)
from flash_attention.flash_attention_v1 import flash_attention


# ═══════════════════════════════════════════════════════════════════
# 嵌入与位置编码（本地定义，各项目实现方式不同，不适合放 lib）
# ═══════════════════════════════════════════════════════════════════

def Embedding(num_embeddings, embedding_dim):
    """Token 嵌入层：将整数 token ID 映射为稠密向量。

    Args:
        num_embeddings: 词表大小
        embedding_dim:  嵌入维度
    """
    def init_fun(rng, input_shape):
        # input_shape: (batch, seq_len) — 整数 token 索引
        output_shape = input_shape + (embedding_dim,)
        emb = jax.random.normal(rng, (num_embeddings, embedding_dim)) * 0.02
        return output_shape, emb

    def apply_fun(params, inputs, **kwargs):
        return jnp.take(params, inputs, axis=0)

    return init_fun, apply_fun


def PositionalEncoding(max_len, embed_dim):
    """可学习的位置编码 — 与 ViT 中一致的做法。

    Args:
        max_len:   位置编码表容纳的最大序列长度
        embed_dim: 嵌入维度
    """
    def init_fun(rng, input_shape):
        pe = jax.random.normal(rng, (1, max_len, embed_dim)) * 0.02
        return input_shape, pe

    def apply_fun(params, inputs, **kwargs):
        return inputs + params[:, :inputs.shape[1], :]

    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# 工具函数：mask 广播
# ═══════════════════════════════════════════════════════════════════

def _safe_block_size(requested, seq_len):
    """将 block_size clamp 到能整除 seq_len 的最接近安全值。

    flash_attention 要求 query/key seq_len 能被 block_size 整除。
    当返回的 block_size < seq_len 时触发分块 (tiled) 模式，否则为单块模式。

    注意：分块模式曾因上游 epsilon bug (stax_plus.py) 导致训练不稳定，
    该 bug 已修复，分块/单块模式现在数值等价。详见 stax_plus.py 文件头注释。
    """
    if requested <= 0 or seq_len <= 0:
        return 1
    bs = min(requested, seq_len)
    while bs > 0 and seq_len % bs != 0:
        bs -= 1
    return max(bs, 1)


def _broadcast_padding_mask(mask, n_heads):
    """将 padding mask 从 [batch, seq] 广播到 [batch * n_heads, seq]

    flash_attention 期望 mask 的 batch 维度已合并 heads。
    mask: [batch, seq], True = 保留（有效位置）
    """
    if mask is None:
        return None
    batch, seq = mask.shape
    # [batch, seq] → [batch, 1, seq] → [batch, n_heads, seq] → [batch*n_heads, seq]
    mask = mask[:, None, :]                # (batch, 1, seq)
    mask = jnp.broadcast_to(mask, (batch, n_heads, seq))
    mask = mask.reshape(batch * n_heads, seq)
    return mask


# ═══════════════════════════════════════════════════════════════════
# 1. 多头自注意力 — Flash Attention
# ═══════════════════════════════════════════════════════════════════

def MultiHeadSelfAttention(n_heads, head_dim, causal=False,
                           block_size_q=1024, block_size_kv=1024):
    """多头自注意力层（Flash Attention），遵循 stax 约定。

    Args:
        n_heads:       注意力头数
        head_dim:      每个头的维度
        causal:        是否使用因果掩码（解码器自注意力需设为 True）
        block_size_q:  Q 分块大小
        block_size_kv: K/V 分块大小

    Returns:
        (init_fun, apply_fun)  — stax 风格的层对
    """
    def init_fun(rng, input_shape):
        batch_size, seq_len, features = input_shape

        k1, k2, k3, k4 = jax.random.split(rng, 4)
        Wq = jax.random.normal(k1, (features, n_heads * head_dim)) * 0.02
        Wk = jax.random.normal(k2, (features, n_heads * head_dim)) * 0.02
        Wv = jax.random.normal(k3, (features, n_heads * head_dim)) * 0.02
        Wo = jax.random.normal(k4, (n_heads * head_dim, features)) * 0.02

        bq = jnp.zeros((n_heads * head_dim,))
        bk = jnp.zeros((n_heads * head_dim,))
        bv = jnp.zeros((n_heads * head_dim,))
        bo = jnp.zeros((features,))

        params = (Wq, bq, Wk, bk, Wv, bv, Wo, bo)
        return input_shape, params

    def apply_fun(params, inputs, padding_mask=None, **kwargs):
        Wq, bq, Wk, bk, Wv, bv, Wo, bo = params

        batch_size, seq_len, features = inputs.shape

        # 线性投影
        Q = jnp.dot(inputs, Wq) + bq  # (batch, seq, n_heads*head_dim)
        K = jnp.dot(inputs, Wk) + bk
        V = jnp.dot(inputs, Wv) + bv

        # 重塑为 (batch, seq, n_heads, head_dim) → (batch, n_heads, seq, head_dim)
        Q = Q.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

        # 合并 batch 和 heads：flash_attention 期望 (batch*n_heads, seq, head_dim)
        Q = Q.reshape(batch_size * n_heads, seq_len, head_dim)
        K = K.reshape(batch_size * n_heads, seq_len, head_dim)
        V = V.reshape(batch_size * n_heads, seq_len, head_dim)

        # mask 广播
        mask = _broadcast_padding_mask(padding_mask, n_heads)

        # Flash Attention — 自动 clamp block_size 到能整除 seq_len 的安全值
        bq = _safe_block_size(block_size_q, seq_len)
        bk = _safe_block_size(block_size_kv, seq_len)
        attn_out = flash_attention(
            Q, K, V,
            padding_mask_k=mask,
            padding_mask_q=mask,
            causal=causal,
            block_size_q=bq,
            block_size_kv=bk,
        )  # (batch*n_heads, seq, head_dim)

        # 拆分 heads: (batch, n_heads, seq, head_dim) → (batch, seq, n_heads, head_dim)
        attn_out = attn_out.reshape(batch_size, n_heads, seq_len, head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3)

        # 合并多头: (batch, seq, n_heads*head_dim)
        attn_out = attn_out.reshape(batch_size, seq_len, n_heads * head_dim)

        # 输出投影
        output = jnp.dot(attn_out, Wo) + bo
        return output

    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# 2. 多头交叉注意力 — Flash Attention
# ═══════════════════════════════════════════════════════════════════

def MultiHeadCrossAttention(n_heads, head_dim,
                            block_size_q=1024, block_size_kv=1024):
    """多头交叉注意力层（Flash Attention），遵循 stax 约定。

    输入为 (query, key, value) 三元组，由 stax.parallel 或手动传入。

    Args:
        n_heads:       注意力头数
        head_dim:      每个头的维度
        block_size_q:  Q 分块大小
        block_size_kv: K/V 分块大小

    Returns:
        (init_fun, apply_fun)
    """
    def init_fun(rng, input_shapes):
        query_shape, key_shape, value_shape = input_shapes
        _, q_seq_len, q_features = query_shape
        _, k_seq_len, k_features = key_shape

        k1, k2, k3, k4 = jax.random.split(rng, 4)
        Wq = jax.random.normal(k1, (q_features, n_heads * head_dim)) * 0.02
        Wk = jax.random.normal(k2, (k_features, n_heads * head_dim)) * 0.02
        Wv = jax.random.normal(k3, (value_shape[-1], n_heads * head_dim)) * 0.02
        Wo = jax.random.normal(k4, (n_heads * head_dim, q_features)) * 0.02

        bq = jnp.zeros((n_heads * head_dim,))
        bk = jnp.zeros((n_heads * head_dim,))
        bv = jnp.zeros((n_heads * head_dim,))
        bo = jnp.zeros((q_features,))

        params = (Wq, bq, Wk, bk, Wv, bv, Wo, bo)
        return query_shape, params

    def apply_fun(params, inputs,
                  padding_mask_q=None, padding_mask_k=None, **kwargs):
        Wq, bq, Wk, bk, Wv, bv, Wo, bo = params
        query, key, value = inputs

        batch_size, q_seq_len, q_features = query.shape
        _, k_seq_len, _ = key.shape

        # 线性投影
        Q = jnp.dot(query, Wq) + bq
        K = jnp.dot(key, Wk) + bk
        V = jnp.dot(value, Wv) + bv

        # 重塑并合并 batch+heads
        Q = Q.reshape(batch_size, q_seq_len, n_heads, head_dim)
        Q = Q.transpose(0, 2, 1, 3).reshape(batch_size * n_heads, q_seq_len, head_dim)

        K = K.reshape(batch_size, k_seq_len, n_heads, head_dim)
        K = K.transpose(0, 2, 1, 3).reshape(batch_size * n_heads, k_seq_len, head_dim)

        V = V.reshape(batch_size, k_seq_len, n_heads, head_dim)
        V = V.transpose(0, 2, 1, 3).reshape(batch_size * n_heads, k_seq_len, head_dim)

        # mask 广播
        mask_q = _broadcast_padding_mask(padding_mask_q, n_heads)
        mask_k = _broadcast_padding_mask(padding_mask_k, n_heads)

        # Flash Attention — 自动 clamp block_size 到能整除 seq_len 的安全值
        bq = _safe_block_size(block_size_q, q_seq_len)
        bk = _safe_block_size(block_size_kv, k_seq_len)
        attn_out = flash_attention(
            Q, K, V,
            padding_mask_k=mask_k,
            padding_mask_q=mask_q,
            causal=False,
            block_size_q=bq,
            block_size_kv=bk,
        )

        # 拆分 heads 并合并
        attn_out = attn_out.reshape(batch_size, n_heads, q_seq_len, head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3)
        attn_out = attn_out.reshape(batch_size, q_seq_len, n_heads * head_dim)

        output = jnp.dot(attn_out, Wo) + bo
        return output

    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# 3. Transformer 编码器块 — Pre-LN 结构
# ═══════════════════════════════════════════════════════════════════

def TransformerEncoderBlock(n_heads, head_dim, embed_dim, mlp_dim,
                            dropout_rate=0.1, block_size=1024):
    """Pre-LN Transformer 编码器块。

    结构: x → LN → Attn → (+x) → LN → FFN → (+x)

    Args:
        n_heads:      注意力头数
        head_dim:     每个头的维度
        embed_dim:    嵌入/模型维度
        mlp_dim:      FFN 中间层维度
        dropout_rate: Dropout 比率
        block_size:   分块大小

    Returns:
        (init_fun, apply_fun)
    """
    # 自注意力 + Dropout
    self_attn = serial(
        MultiHeadSelfAttention(
            n_heads, head_dim, causal=False,
            block_size_q=block_size, block_size_kv=block_size,
        ),
        Dropout(dropout_rate),
    )

    # FFN + Dropout
    ff_net = serial(
        Dense(mlp_dim),
        Gelu,
        Dropout(dropout_rate),
        Dense(embed_dim),
        Dropout(dropout_rate),
    )

    def init_fun(rng, input_shape):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        # 自注意力路径（含 LayerNorm）
        _, attn_params, attn_bn = self_attn[0](rng1, input_shape)
        _, ln1_params = LayerNorm()[0](rng2, input_shape)
        # FFN 路径（含 LayerNorm）
        _, ff_params, ff_bn = ff_net[0](rng3, input_shape)
        _, ln2_params = LayerNorm()[0](rng4, input_shape)

        params = (attn_params, attn_bn, ff_params, ff_bn, ln1_params, ln2_params)
        return input_shape, params

    def apply_fun(params, inputs, padding_mask=None, rng=None, **kwargs):
        attn_params, attn_bn, ff_params, ff_bn, ln1_params, ln2_params = params

        # --- 自注意力子层（带残差）---
        x = LayerNorm()[1](ln1_params, inputs)
        rng1, rng = _split_rng(rng)
        attn_out, _ = self_attn[1](
            attn_params, attn_bn, x, padding_mask=padding_mask, rng=rng1, **kwargs
        )
        x = inputs + attn_out

        # --- FFN 子层（带残差）---
        residual = x
        x = LayerNorm()[1](ln2_params, x)
        rng2, rng = _split_rng(rng)
        ff_out, _ = ff_net[1](ff_params, ff_bn, x, rng=rng2, **kwargs)
        x = residual + ff_out

        return x

    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# 4. Transformer 解码器块 — Pre-LN 结构
# ═══════════════════════════════════════════════════════════════════

def TransformerDecoderBlock(n_heads, head_dim, embed_dim, mlp_dim,
                            dropout_rate=0.1, block_size=1024):
    """Pre-LN Transformer 解码器块。

    结构:
        x → LN → SelfAttn(causal) → (+x)
          → LN → CrossAttn(enc_out) → (+x)
          → LN → FFN → (+x)

    Args:
        n_heads:      注意力头数
        head_dim:     每个头的维度
        embed_dim:    嵌入/模型维度
        mlp_dim:      FFN 中间层维度
        dropout_rate: Dropout 比率
        block_size:   分块大小

    Returns:
        (init_fun, apply_fun)
    """
    # 因果自注意力
    causal_attn = serial(
        MultiHeadSelfAttention(
            n_heads, head_dim, causal=True,
            block_size_q=block_size, block_size_kv=block_size,
        ),
        Dropout(dropout_rate),
    )

    # 交叉注意力
    cross_attn = serial(
        MultiHeadCrossAttention(
            n_heads, head_dim,
            block_size_q=block_size, block_size_kv=block_size,
        ),
        Dropout(dropout_rate),
    )

    # FFN
    ff_net = serial(
        Dense(mlp_dim),
        Gelu,
        Dropout(dropout_rate),
        Dense(embed_dim),
        Dropout(dropout_rate),
    )

    def init_fun(rng, input_shape):
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        _, attn_params, attn_bn = causal_attn[0](rng1, input_shape)

        # 交叉注意力：假设编码器输出形状与输入相同
        enc_shape = input_shape
        _, cross_params, cross_bn = cross_attn[0](
            rng2, (input_shape, enc_shape, enc_shape)
        )

        _, ff_params, ff_bn = ff_net[0](rng3, input_shape)

        # 3 个 LayerNorm
        ln_rngs = jax.random.split(rng3, 3)
        ln_params = []
        for i in range(3):
            _, ln_p = LayerNorm()[0](ln_rngs[i], input_shape)
            ln_params.append(ln_p)

        params = (
            attn_params, attn_bn, cross_params, cross_bn, ff_params, ff_bn,
            ln_params[0], ln_params[1], ln_params[2],
        )
        return input_shape, params

    def apply_fun(params, inputs, encoder_output=None,
                  tgt_padding_mask=None, src_padding_mask=None,
                  rng=None, **kwargs):
        (attn_params, attn_bn, cross_params, cross_bn, ff_params, ff_bn,
         ln1, ln2, ln3) = params

        # --- 因果自注意力子层 ---
        x = LayerNorm()[1](ln1, inputs)
        rng1, rng = _split_rng(rng)
        attn_out, _ = causal_attn[1](
            attn_params, attn_bn, x,
            padding_mask=tgt_padding_mask, rng=rng1, **kwargs,
        )
        x = inputs + attn_out

        # --- 交叉注意力子层 ---
        residual = x
        x = LayerNorm()[1](ln2, x)
        rng2, rng = _split_rng(rng)
        cross_out, _ = cross_attn[1](
            cross_params, cross_bn, (x, encoder_output, encoder_output),
            padding_mask_q=tgt_padding_mask,
            padding_mask_k=src_padding_mask,
            rng=rng2, **kwargs,
        )
        x = residual + cross_out

        # --- FFN 子层 ---
        residual = x
        x = LayerNorm()[1](ln3, x)
        rng3, rng = _split_rng(rng)
        ff_out, _ = ff_net[1](ff_params, ff_bn, x, rng=rng3, **kwargs)
        x = residual + ff_out

        return x

    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# 5. 高层封装：Encoder / Decoder / Transformer
# ═══════════════════════════════════════════════════════════════════

def TransformerEncoder(num_layers, n_heads, head_dim, embed_dim, mlp_dim,
                       dropout_rate=0.1, block_size=1024):
    """堆叠多个编码器块 + 最终 LayerNorm。

    Args:
        num_layers:   编码器块数量
        n_heads:      注意力头数
        head_dim:     每个头的维度
        embed_dim:    嵌入维度
        mlp_dim:      FFN 中间层维度
        dropout_rate: Dropout 比率
        block_size:   分块大小

    Returns:
        (init_fun, apply_fun)
    """
    layers = [
        TransformerEncoderBlock(
            n_heads, head_dim, embed_dim, mlp_dim,
            dropout_rate, block_size,
        )
        for _ in range(num_layers)
    ]
    return serial(*layers, LayerNorm())


def TransformerDecoder(num_layers, n_heads, head_dim, embed_dim, mlp_dim,
                       dropout_rate=0.1, block_size=1024):
    """堆叠多个解码器块 + 最终 LayerNorm。

    Args:
        num_layers:   解码器块数量
        n_heads:      注意力头数
        head_dim:     每个头的维度
        embed_dim:    嵌入维度
        mlp_dim:      FFN 中间层维度
        dropout_rate: Dropout 比率
        block_size:   分块大小

    Returns:
        (init_fun, apply_fun)
    """
    layers = [
        TransformerDecoderBlock(
            n_heads, head_dim, embed_dim, mlp_dim,
            dropout_rate, block_size,
        )
        for _ in range(num_layers)
    ]
    return serial(*layers, LayerNorm())


def Transformer(src_vocab_size, tgt_vocab_size,
                embed_dim, n_heads, head_dim, mlp_dim,
                num_encoder_layers, num_decoder_layers,
                max_len, dropout_rate=0.1, block_size=1024):
    """完整的 Encoder-Decoder Transformer 模型。

    Args:
        src_vocab_size:       源词汇表大小
        tgt_vocab_size:       目标词汇表大小
        embed_dim:            嵌入维度
        n_heads:              注意力头数
        head_dim:             每个头的维度
        mlp_dim:              FFN 中间层维度
        num_encoder_layers:   编码器层数
        num_decoder_layers:   解码器层数
        max_len:              最大序列长度
        dropout_rate:         Dropout 比率
        block_size:           分块大小

    Returns:
        (init_fun, apply_fun)
    """
    # 编码器：Embed → PosEnc → EncBlocks → LayerNorm
    encoder = serial(
        Embedding(src_vocab_size, embed_dim),
        PositionalEncoding(max_len, embed_dim),
        TransformerEncoder(
            num_encoder_layers, n_heads, head_dim, embed_dim, mlp_dim,
            dropout_rate, block_size,
        ),
    )

    # 解码器：Embed → PosEnc → DecBlocks → LayerNorm → Dense → LogSoftmax
    decoder = serial(
        Embedding(tgt_vocab_size, embed_dim),
        PositionalEncoding(max_len, embed_dim),
        TransformerDecoder(
            num_decoder_layers, n_heads, head_dim, embed_dim, mlp_dim,
            dropout_rate, block_size,
        ),
        Dense(tgt_vocab_size),
        stax.LogSoftmax,
    )

    def init_fun(rng, input_shapes):
        src_shape, tgt_shape = input_shapes
        rng1, rng2 = jax.random.split(rng)
        _, enc_params, enc_bn = encoder[0](rng1, src_shape)
        _, dec_params, dec_bn = decoder[0](rng2, tgt_shape)
        params = (enc_params, enc_bn, dec_params, dec_bn)
        return (src_shape, tgt_shape), params

    @jax.jit
    def apply_fun(params, inputs,
                  src_padding_mask=None,
                  tgt_padding_mask=None,
                  rng=None, **kwargs):
        enc_params, enc_bn, dec_params, dec_bn = params
        src_inputs, tgt_inputs = inputs

        # 编码器
        enc_output, _ = encoder[1](
            enc_params, enc_bn, src_inputs,
            padding_mask=src_padding_mask,
            rng=rng, **kwargs,
        )

        # 解码器（需要 encoder_output 作为交叉注意力输入）
        dec_output, _ = decoder[1](
            dec_params, dec_bn, tgt_inputs,
            encoder_output=enc_output,
            tgt_padding_mask=tgt_padding_mask,
            src_padding_mask=src_padding_mask,
            rng=rng, **kwargs,
        )

        return dec_output

    @jax.jit
    def encode_fn(params, src_inputs, src_padding_mask=None):
        """编码器前向传播（推理用，无 dropout，可缓存结果）"""
        enc_params, enc_bn, _, _ = params
        enc_output, _ = encoder[1](
            enc_params, enc_bn, src_inputs,
            padding_mask=src_padding_mask,
            is_training=False,
        )
        return enc_output

    @jax.jit
    def decode_fn(params, tgt_inputs, encoder_output,
                  src_padding_mask=None, tgt_padding_mask=None):
        """解码器前向传播（推理用，使用缓存的编码器输出）"""
        _, _, dec_params, dec_bn = params
        dec_output, _ = decoder[1](
            dec_params, dec_bn, tgt_inputs,
            encoder_output=encoder_output,
            tgt_padding_mask=tgt_padding_mask,
            src_padding_mask=src_padding_mask,
            is_training=False,
        )
        return dec_output

    return init_fun, apply_fun, encode_fn, decode_fn


# ═══════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════

def _split_rng(rng):
    """安全拆分 rng，兼容 rng=None（eval 模式 dropout 自动关闭）的情况。

    用于 TransformerEncoderBlock / TransformerDecoderBlock 内部：
    将同一个 rng 拆分为 attention 子层和 FFN 子层的独立 dropout rng。
    """
    if rng is not None:
        return jax.random.split(rng)
    return None, None


# ═══════════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time

    n_heads = 8
    head_dim = 64
    embed_dim = n_heads * head_dim  # 512
    hidden_dim = 2048
    num_layers = 6
    max_len = 512
    batch_size = 2

    print("=" * 60)
    print("Flash Attention Transformer — 模块测试")
    print("=" * 60)

    # ---- 测试 Encoder ----
    print("\n[1] 测试 TransformerEncoder...")
    enc_init, enc_apply = TransformerEncoder(
        num_layers=2, n_heads=n_heads, head_dim=head_dim,
        embed_dim=embed_dim, mlp_dim=hidden_dim,
        dropout_rate=0.1, block_size=128,
    )

    rng = jax.random.PRNGKey(42)
    input_shape = (batch_size, max_len, embed_dim)
    _, enc_params, enc_bn = enc_init(rng, input_shape)

    x = jax.random.normal(rng, input_shape)
    fwd_rng, rng = jax.random.split(rng)
    enc_out, _ = enc_apply(enc_params, enc_bn, x, rng=fwd_rng)
    print(f"   输入形状: {x.shape} → 输出形状: {enc_out.shape} ✓")

    # ---- 测试 Decoder ----
    print("\n[2] 测试 TransformerDecoder...")
    dec_init, dec_apply = TransformerDecoder(
        num_layers=2, n_heads=n_heads, head_dim=head_dim,
        embed_dim=embed_dim, mlp_dim=hidden_dim,
        dropout_rate=0.1, block_size=128,
    )

    rng, init_rng = jax.random.split(rng)
    _, dec_params, dec_bn = dec_init(init_rng, input_shape)

    # decoder_apply 需要 encoder_output 和 padding_masks
    fwd_rng, rng = jax.random.split(rng)
    dec_out, _ = dec_apply(
        dec_params, dec_bn, x,
        encoder_output=enc_out,
        rng=fwd_rng,
    )
    print(f"   输入形状: {x.shape} → 输出形状: {dec_out.shape} ✓")

    # ---- 测试完整 Transformer ----
    print("\n[3] 测试完整 Transformer (翻译模型)...")
    src_vocab, tgt_vocab = 1000, 1000

    model_init, model_apply, model_encode, model_decode = Transformer(
        src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab,
        embed_dim=embed_dim, n_heads=n_heads, head_dim=head_dim,
        mlp_dim=hidden_dim,
        num_encoder_layers=3, num_decoder_layers=3,
        max_len=max_len, dropout_rate=0.1, block_size=128,
    )

    rng, init_rng = jax.random.split(rng)
    src_shape = (batch_size, max_len)
    tgt_shape = (batch_size, max_len)
    out_shape, params = model_init(init_rng, (src_shape, tgt_shape))

    src_tokens = jax.random.randint(rng, src_shape, 0, src_vocab)
    tgt_tokens = jax.random.randint(rng, tgt_shape, 0, tgt_vocab)

    rng, fwd_rng = jax.random.split(rng)
    output = model_apply(params, (src_tokens, tgt_tokens), rng=fwd_rng)
    print(f"   输入: src{src_shape}, tgt{tgt_shape} → 输出: {output.shape} ✓")

    # 参数统计
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\n   模型参数总量: {total_params:,}")

    # 测试 encode_fn / decode_fn (推理模式)
    print("\n[3b] 测试 encode_fn / decode_fn (推理模式)...")
    enc_out = model_encode(params, src_tokens)
    dec_out = model_decode(params, tgt_tokens, enc_out)
    print(f"   encode: {enc_out.shape}, decode: {dec_out.shape} ✓")

    # ---- 测试自注意力 (供 ViT 等模块直接使用) ----
    print("\n[4] 测试 MultiHeadSelfAttention (ViT 风格)...")
    # ViT 输入: (batch, num_patches+1, embed_dim) — 如 CIFAR patch=4 得 64+1=65
    # 注：seq_len 必须能被 block_size_q 整除
    attn_init, attn_apply = MultiHeadSelfAttention(
        n_heads=6, head_dim=64, causal=False,
        block_size_q=65, block_size_kv=65,
    )

    vit_shape = (4, 65, 384)
    rng, init_rng = jax.random.split(rng)
    _, attn_params = attn_init(init_rng, vit_shape)
    vit_x = jax.random.normal(rng, vit_shape)
    vit_out = attn_apply(attn_params, vit_x)
    print(f"   ViT 输入: {vit_shape} → 输出: {vit_out.shape} ✓")

    # 测试带 padding_mask 的版本
    print("\n[5] 测试 MultiHeadSelfAttention + padding_mask...")
    mask = jnp.ones((4, 65), dtype=bool)
    mask = mask.at[:, -5:].set(False)  # 最后 5 个位置 masked
    vit_out_masked = attn_apply(attn_params, vit_x, padding_mask=mask)
    print(f"   带 mask 输出: {vit_out_masked.shape} ✓")

    print("\n" + "=" * 60)
    print("所有模块测试通过 ✓")
    print("=" * 60)
