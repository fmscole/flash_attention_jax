# Flash Attention JAX

Flash Attention（v1 和 v2）的 JAX 实现，通过自定义 VJP 实现内存高效精确注意力。在此基础上构建了 Transformer 组件库、CIFAR-10 视觉变压器（ViT）和英中翻译模型。

## 安装

```bash
pip install jax optax torch torchvision numpy matplotlib jieba opencc
```

无需安装包，直接从 `src/` 导入：

```python
import sys
sys.path.insert(0, 'src')
from flash_attention import flash_attention, flash_attention_v2
```

## Flash Attention

核心模块提供两种分块注意力变体，均使用手动 VJP（自定义梯度）实现，避免显存化完整的 `seq_len × seq_len` 注意力矩阵。

### API

两个函数共享相同签名：

```
flash_attention(q, k, v, padding_mask_k=None, padding_mask_q=None,
                causal=False, block_size_q=128, block_size_kv=128)
flash_attention_v2(q, k, v, padding_mask_k=None, padding_mask_q=None,
                   causal=False, block_size_q=128, block_size_kv=128)
```

**输入：**

| 参数 | 形状 | 说明 |
|-----------|-------|-------------|
| `q` | `[batch_heads, seq_len, head_dim]` | Query。`batch_heads = batch_size × num_heads` |
| `k` | `[batch_heads, kv_seq_len, head_dim]` | Key |
| `v` | `[batch_heads, kv_seq_len, head_dim]` | Value |
| `padding_mask_k` | `[batch_heads, kv_seq_len]` 或 `None` | 布尔值掩码。`True` = 有效，`False` = 填充 |
| `padding_mask_q` | `[batch_heads, seq_len]` 或 `None` | 同上 |
| `causal` | `bool` | 为 `True` 时施加因果掩码 |
| `block_size_q` | `int` | Query 维度的分块大小 |
| `block_size_kv` | `int` | Key/Value 维度的分块大小 |

**返回：** `[batch_heads, seq_len, head_dim]`

**注意：** 输入使用 `[batch_heads, ...]` 格式（batch 和 head 合并为一个维度）。调用前需用 `q.reshape(batch * n_heads, seq, d)` 转换。

### v1 与 v2 对比

| | v1 (`flash_attention`) | v2 (`flash_attention_v2`) |
|---|---|---|
| **前向循环** | Q-block 外层, KV-block 内层 | KV-block 外层, Q-block 内层 |
| **反向循环** | 同前向（Q-外, KV-内） | 同前向（KV-外, Q-内） |
| **参考论文** | 标准 Flash Attention | FlashAttention-2 (Dao 2023) §3.2 |
| **数值** | 与 v2 一致（差异 < 1e-5） | 与 v1 一致 |

v2 的 KV-外反向循环相比 v1 减少了 SRAM 写入次数，对应 FlashAttention-2 论文中的序列长度并行化优化。两者前向和梯度结果完全一致。

```python
from flash_attention import flash_attention, flash_attention_v2

# 调用方式相同
out1 = flash_attention(q, k, v, causal=True, block_size_q=128, block_size_kv=128)
out2 = flash_attention_v2(q, k, v, causal=True, block_size_q=128, block_size_kv=128)
assert jnp.allclose(out1, out2, atol=1e-5)
```

### 数值稳定性

当 `block_size < seq_len` 时，注意力按分块计算。归一化使用统一 epsilon（`1e-12`），前向 softmax 和反向 log-sum-exp 恢复共用同一值，确保反向 `exp(S - L)` 精确恢复前向权重，消除长序列训练时的梯度漂移问题。

### 运行测试

```bash
# 单元测试覆盖 v1 和 v2（自包含，无需额外依赖）
python src/flash_attention/flash_attention_v1.py
python src/flash_attention/flash_attention_v2.py

# v1 专用测试套件
python src/flash_attention/flash_attention_test.py
```

每个文件在主函数中包含测试类，覆盖：前向/反向（无掩码、有掩码、因果模式）、数值稳定性、不同分块大小。

## Transformers

基于 Flash Attention 的 Transformer 组件库。两份并行实现：`transformer_flash_v1`（使用 `flash_attention`）和 `transformer_flash_v2`（使用 `flash_attention_v2`），API 完全相同，输出可互换。

### 组件

```python
from transformers.transformer_flash_v1 import (
    Embedding, PositionalEncoding,
    MultiHeadSelfAttention, MultiHeadCrossAttention,
    TransformerEncoderBlock, TransformerDecoderBlock,
    TransformerEncoder, TransformerDecoder,
    Transformer,  # 完整 encoder-decoder 模型
)
```

| 组件 | 说明 |
|-----------|-------------|
| `Embedding(num_embeddings, dim)` | Token 嵌入，基于 `jnp.take` |
| `PositionalEncoding(max_len, dim)` | 可学习位置编码 |
| `MultiHeadSelfAttention(n_heads, head_dim, causal, block_size_q, block_size_kv)` | QKV 投影 → flash attention → 输出投影 |
| `MultiHeadCrossAttention(n_heads, head_dim, ...)` | 交叉注意力，Q 与 K/V 分离 |
| `TransformerEncoderBlock(n_heads, head_dim, embed_dim, mlp_dim, dropout_rate)` | Pre-LN：自注意力 → FFN，均带残差连接 |
| `TransformerDecoderBlock(n_heads, head_dim, embed_dim, mlp_dim, dropout_rate)` | Pre-LN：因果自注意力 → 交叉注意力 → FFN |
| `TransformerEncoder(num_layers, ...)` | 堆叠编码器块 + 最终 LayerNorm |
| `TransformerDecoder(num_layers, ...)` | 堆叠解码器块 + 最终 LayerNorm |
| `Transformer(src_vocab_size, tgt_vocab_size, embed_dim, n_heads, head_dim, mlp_dim, num_encoder_layers, num_decoder_layers, max_len, block_size)` | 完整 encoder-decoder。返回 `(init, apply, encode, decode)` |

所有 Transformer 模块采用 **Pre-LN** 架构。Dropout 通过 `apply` 的 `is_training` 关键字参数控制。

### 形状约定

Transformer 模块内部自动处理 batch 和 head 的展平与恢复：

```python
# 输入：[batch, seq_len]
# Transformer 内部完成 [batch, seq_len] → [batch_heads, seq, head_dim] 的转换
```

### MAE 工具

```python
from transformers.mae_utils import random_masking, unshuffle
```

| 函数 | 用途 |
|----------|---------|
| `random_masking(x, mask_ratio, rng)` → `(x_visible, ids_restore, mask)` | 随机掩码 patches，用于 MAE 预训练 |
| `unshuffle(x_visible, ids_restore, mask_token, num_patches)` | 将编码后的可见 patches 恢复为完整序列，掩码位置填充为 mask token |

### 层库 (`lib.stax_plus`)

基于 JAX stax 的扩展层库，Transformer 模块内部使用：

```python
from lib.stax_plus import Dense, Conv, LayerNorm, BatchNorm, Dropout, Gelu, serial, parallel
```

`serial()` / `parallel()` 组合子感知状态变化，自动传递 BatchNorm state 并处理初始化的三值返回。

## ViT：Vision Transformer（CIFAR-10）

基于 Vision Transformer 的 CIFAR-10 分类模型。

```
src/vit/
└── vit_cifar10_stax.py              # 端到端监督训练
```

### 用法

```bash
python src/vit/vit_cifar10_stax.py
```

### 模型架构

| 组件 | 配置 |
|-----------|--------|
| Patch 嵌入 | Conv 4×4, stride 4 → 64 patches |
| 嵌入维度 | 384 |
| 头数 | 6 |
| 层数 | 7 |
| MLP 维度 | 768 |
| 位置编码 | 可学习（非正弦固定） |
| 注意力 | Flash Attention v1 |

### 训练细节

| 配置 | 值 |
|--------|-------|
| 轮次 | 500 |
| 峰值学习率 | 3e-4, warmup 10 轮 + cosine decay |
| 批大小 | 128 |
| 优化器 | AdamW（weight_decay=0.1） |
| 数据增强 | RandAugment + MixUp（α=0.8） |
| 标签平滑 | 0.1 |
| 准确率 | 90~91% |

## Translation：英中翻译

基于 Encoder-Decoder Transformer 的英→中翻译模型，使用 AiChallenger 机器翻译数据集（约 1000 万句对）。

```
src/translation/
├── prepare_ai_challenger.py              # 数据集预处理
└── translate_stax_flash.py               # 端到端训练
```

### 数据集准备

```bash
# 从原始 AiChallenger 文件生成训练用 TSV
python src/translation/prepare_ai_challenger.py
```

需先下载 [AiChallenger 机器翻译数据集](https://aistudio.baidu.com/datasetdetail/220848)。

### 用法

```bash
# 在平行语料上从头训练
python src/translation/translate_stax_flash.py
```

### 模型架构

| 组件 | 配置 |
|-----------|--------|
| Encoder / Decoder | 各 6 层，Pre-LN |
| 嵌入维度 | 512（8 头 × 64） |
| FFN 维度 | 2048 |
| 词表 | ~55K（基于词频，非 BPE） |
| 最大长度 | 64 |
| 注意力 | Flash Attention v1 |
| 推理 | 贪心解码 |

### 训练细节

| 配置 | 值 |
|--------|-------|
| 轮次 | 10000（早停触发终止） |
| 峰值学习率 | 3e-5, warmup 4000 steps, cosine decay → 1e-5 |
| 批大小 | 64 |
| 优化器 | AdamW（weight_decay=1e-4） |
| 梯度裁剪 | global norm 1.0 |
| 早停 | val loss 连续 3 次验证不降 |
| 推理 | 贪心解码，遇 `<eos>` 停止 |

## License

MIT
