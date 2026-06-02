"""Flash Attention with custom VJP for JAX.

IO-aware exact attention with tiled forward/backward passes.
Supports causal masking and padding masks.

Input shapes:
    q, k, v: [batch_heads, seq_len, head_dim]  (batch_heads = batch_size * num_heads)
    padding_mask_k, padding_mask_q: [batch_heads, seq_len] (True = keep)

【数值稳定性修复 — 2024】
分块模式下 (block_size < seq_len)，前向归一化和 log_sum_exp 保存必须使用同一个
epsilon，否则反向传播 exp(S - log_sum_exp) 恢复的 attention 权重与前向不一致，
在多 epoch 训练中累积梯度漂移导致崩塌。修复方案：统一使用 _MIN_NORMALIZER=1e-12。
详见下方 _MIN_NORMALIZER 注释块。
"""

import unittest
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

MAX_EXP = 50.0

# ═══════════════════════════════════════════════════════════════════════════════
# flash_attention 数值一致性 — 消除分块模式下的前向/反向误差累积
# ═══════════════════════════════════════════════════════════════════════════════
#
# 【Bug 根因】
# online-softmax 分块计算中：
# 前向: output = ... / max(global_sum_exp, ε)            … ① 用 ε=1e-6
# 前向: L = global_max + log(global_sum_exp + ε')       … ② 用 ε'=1e-6
# 反向: weights = exp(S - L) = exp(S-max) / (sum_exp + ε') … ③ 用 ε'=1e-6
# 问题: ① 和 ③ 的分母不同（max(s, ε) vs s + ε'），微小偏差在多层×多tile
#       训练中累积导致梯度漂移，最终触发训练崩塌 (如 ViT CIFAR-100 epoch 11/38 崩塌)。
#
# 【修复】统一两处 epsilon 为 _MIN_NORMALIZER=1e-12
# 前向: output = ... / max(sum_exp, 1e-12)
# 前向: L = max + log(max(sum_exp, 1e-12))
# 反向: exp(S - L) = exp(S-max) / max(sum_exp, 1e-12)  ← 与①精确一致 ✓
#
# 【安全性】sum_exp ≥ 1.0 恒成立 → 相对误差 1e-12（float32 精度 ~1.2e-7，差5个量级）
# ═══════════════════════════════════════════════════════════════════════════════
_MIN_NORMALIZER = 1e-12


def _make_attention_mask(causal, query_start_idx, query_block_size,
                          kv_start_idx, kv_block_size,
                          query_mask_block, key_mask_block):
    """Combined attention mask for a Q-K block pair, shared by forward/backward."""

    def _causal_branch():
        query_indices = jnp.arange(query_block_size) + query_start_idx
        key_indices = jnp.arange(kv_block_size) + kv_start_idx
        causal_mask = (query_indices[:, None] >= key_indices[None, :])
        causal_mask = causal_mask[None, :, :]
        return causal_mask & query_mask_block[:, :, None] & key_mask_block[:, None, :]

    def _padding_branch():
        return query_mask_block[:, :, None] & key_mask_block[:, None, :]

    return lax.cond(causal, _causal_branch, _padding_branch)


@partial(jax.jit, static_argnames=['block_size_q', 'block_size_kv'])
def flash_attention_forward(query, key, value, key_padding_mask,
                             query_padding_mask, causal, block_size_q, block_size_kv):
    """Tiled FlashAttention forward pass."""
    batch_heads, query_seq_len, head_dim = query.shape
    _, key_seq_len, value_dim = value.shape

    # Mask defaults
    if key_padding_mask is None:
        key_padding_mask = jnp.ones((batch_heads, key_seq_len), dtype=bool)
    if query_padding_mask is None:
        query_padding_mask = jnp.ones((batch_heads, query_seq_len), dtype=bool)

    # Divisibility check — ValueError survives `python -O`, unlike assert.
    if query_seq_len % block_size_q != 0:
        raise ValueError(
            f"query_seq_len ({query_seq_len}) must be divisible by block_size_q ({block_size_q})")
    if key_seq_len % block_size_kv != 0:
        raise ValueError(
            f"key_seq_len ({key_seq_len}) must be divisible by block_size_kv ({block_size_kv})")

    query_block_size = min(block_size_q, query_seq_len)
    kv_block_size = min(block_size_kv, key_seq_len)
    num_query_blocks = query_seq_len // query_block_size
    num_kv_blocks = key_seq_len // kv_block_size

    scale = 1.0 / jnp.sqrt(head_dim)

    # Use value_dim (not head_dim) for output — correctness under multi-query attention.
    output = jnp.zeros((batch_heads, query_seq_len, value_dim), dtype=query.dtype)
    max_values = jnp.full((batch_heads, query_seq_len), -jnp.inf, dtype=jnp.float32)
    sum_exp_values = jnp.zeros((batch_heads, query_seq_len), dtype=jnp.float32)
    has_valid_global = jnp.zeros((batch_heads, query_seq_len), dtype=bool)

    # ---- KV-block loop (inner) ----
    def kv_block_loop(kv_block_idx, carry):
        block_output, block_sum_exp, block_max, block_query, \
            query_start_idx, query_mask_block, has_valid = carry
        kv_start_idx = kv_block_idx * kv_block_size

        key_block = lax.dynamic_slice_in_dim(key, kv_start_idx, kv_block_size, axis=1)
        value_block = lax.dynamic_slice_in_dim(value, kv_start_idx, kv_block_size, axis=1)
        key_mask_block = lax.dynamic_slice_in_dim(key_padding_mask, kv_start_idx, kv_block_size, axis=1)

        attention_mask = _make_attention_mask(
            causal, query_start_idx, query_block_size,
            kv_start_idx, kv_block_size,
            query_mask_block, key_mask_block)

        attention_scores = jnp.matmul(block_query, key_block.swapaxes(-1, -2)) * scale
        attention_scores = jnp.clip(attention_scores, -MAX_EXP, MAX_EXP)
        attention_scores = jnp.where(attention_mask, attention_scores, -MAX_EXP)

        has_valid = has_valid | jnp.any(attention_mask, axis=-1)

        new_block_max = jnp.maximum(block_max[..., None],
                                     jnp.max(attention_scores, axis=-1, keepdims=True))
        attention_weights = jnp.exp(attention_scores - new_block_max)

        exp_factor = jnp.exp(block_max[..., None] - new_block_max)
        new_sum_exp = exp_factor * block_sum_exp[..., None] + \
                      jnp.sum(attention_weights, axis=-1, keepdims=True)
        new_block_output = exp_factor * block_output + \
                           jnp.matmul(attention_weights, value_block)

        new_block_max = jnp.squeeze(new_block_max, axis=-1)
        new_block_sum_exp = jnp.squeeze(new_sum_exp, axis=-1)

        return (new_block_output, new_block_sum_exp, new_block_max,
                block_query, query_start_idx, query_mask_block, has_valid)

    # ---- Q-block loop (outer) ----
    def query_block_loop(query_block_idx, state):
        output, sum_exp, max_vals, has_valid_global = state
        query_start_idx = query_block_idx * query_block_size

        query_block = lax.dynamic_slice_in_dim(query, query_start_idx, query_block_size, axis=1)
        # query_mask_block sliced once per Q-block (lifted from inner loop).
        query_mask_block = lax.dynamic_slice_in_dim(query_padding_mask, query_start_idx,
                                                     query_block_size, axis=1)

        block_output = jnp.zeros((batch_heads, query_block_size, value_dim), dtype=query.dtype)
        block_sum_exp = jnp.zeros((batch_heads, query_block_size), dtype=jnp.float32)
        block_max = jnp.full((batch_heads, query_block_size), -jnp.inf, dtype=jnp.float32)
        has_valid_block = jnp.zeros((batch_heads, query_block_size), dtype=bool)

        carry_init = (block_output, block_sum_exp, block_max, query_block,
                      query_start_idx, query_mask_block, has_valid_block)
        block_output, block_sum_exp, block_max, _, _, _, has_valid_block = \
            lax.fori_loop(0, num_kv_blocks, kv_block_loop, carry_init)

        # Normalize — 必须与下方 log_sum_exp 使用同一个 _MIN_NORMALIZER，
        # 否则反向 exp(S - L) 恢复的权重与前向不一致（详见文件头注释）。
        block_output = block_output / jnp.maximum(block_sum_exp[..., None], _MIN_NORMALIZER)

        output = lax.dynamic_update_slice_in_dim(output, block_output, query_start_idx, axis=1)
        sum_exp = lax.dynamic_update_slice_in_dim(sum_exp, block_sum_exp, query_start_idx, axis=1)
        max_vals = lax.dynamic_update_slice_in_dim(max_vals, block_max, query_start_idx, axis=1)
        has_valid_global = lax.dynamic_update_slice_in_dim(has_valid_global, has_valid_block,
                                                           query_start_idx, axis=1)
        return (output, sum_exp, max_vals, has_valid_global)

    init_state = (output, sum_exp_values, max_values, has_valid_global)
    output, sum_exp_values, max_values, has_valid_global = \
        lax.fori_loop(0, num_query_blocks, query_block_loop, init_state)

    # Zero out output for fully-masked query positions (no valid keys → output = 0).
    output = jnp.where(has_valid_global[..., None], output, 0.0)

    # log_sum_exp 必须与上方归一化使用同一个 _MIN_NORMALIZER（详见文件头注释）。
    log_sum_exp = jnp.where(jnp.isinf(max_values),
                            max_values,
                            max_values + jnp.log(jnp.maximum(sum_exp_values, _MIN_NORMALIZER)))

    residuals = (query, key, value, log_sum_exp, output, causal,
                 key_padding_mask, query_padding_mask, has_valid_global)
    return output, residuals


@partial(jax.jit, static_argnames=['block_size_q', 'block_size_kv'])
def flash_attention_backward(block_size_q, block_size_kv, residuals, grad_input):
    """Tiled FlashAttention backward pass."""
    query, key, value, log_sum_exp, output, causal, \
        key_padding_mask, query_padding_mask, has_valid_global = residuals

    batch_heads, query_seq_len, head_dim = query.shape
    _, key_seq_len, value_dim = value.shape

    if key_padding_mask is None:
        key_padding_mask = jnp.ones((batch_heads, key_seq_len), dtype=bool)
    if query_padding_mask is None:
        query_padding_mask = jnp.ones((batch_heads, query_seq_len), dtype=bool)

    scale = 1.0 / jnp.sqrt(head_dim)
    output_dot_grad = jnp.sum(output * grad_input, axis=-1)

    query_block_size = min(block_size_q, query_seq_len)
    kv_block_size = min(block_size_kv, key_seq_len)
    num_query_blocks = query_seq_len // query_block_size
    num_kv_blocks = key_seq_len // kv_block_size

    grad_query = jnp.zeros_like(query)
    grad_key = jnp.zeros_like(key)
    grad_value = jnp.zeros_like(value)

    def query_block_loop(query_block_idx, carry):
        grad_query, grad_key, grad_value = carry
        query_start_idx = query_block_idx * query_block_size

        query_block = lax.dynamic_slice_in_dim(query, query_start_idx, query_block_size, axis=1)
        grad_input_block = lax.dynamic_slice_in_dim(grad_input, query_start_idx, query_block_size, axis=1)
        log_sum_exp_block = lax.dynamic_slice_in_dim(log_sum_exp, query_start_idx, query_block_size, axis=1)
        output_dot_grad_block = lax.dynamic_slice_in_dim(output_dot_grad, query_start_idx, query_block_size, axis=1)
        has_valid_block = lax.dynamic_slice_in_dim(has_valid_global, query_start_idx, query_block_size, axis=1)
        query_mask_block = lax.dynamic_slice_in_dim(query_padding_mask, query_start_idx, query_block_size, axis=1)

        log_sum_exp_block = log_sum_exp_block[..., None]
        output_dot_grad_block = output_dot_grad_block[..., None]

        grad_query_block = jnp.zeros_like(query_block)

        def kv_block_loop(kv_block_idx, inner_carry):
            grad_query_block, grad_key, grad_value = inner_carry
            kv_start_idx = kv_block_idx * kv_block_size

            key_block = lax.dynamic_slice_in_dim(key, kv_start_idx, kv_block_size, axis=1)
            value_block = lax.dynamic_slice_in_dim(value, kv_start_idx, kv_block_size, axis=1)
            key_mask_block = lax.dynamic_slice_in_dim(key_padding_mask, kv_start_idx, kv_block_size, axis=1)

            attention_mask = _make_attention_mask(
                causal, query_start_idx, query_block_size,
                kv_start_idx, kv_block_size,
                query_mask_block, key_mask_block)

            attention_scores = jnp.matmul(query_block, key_block.swapaxes(-1, -2)) * scale
            attention_scores = jnp.clip(attention_scores, -MAX_EXP, MAX_EXP)
            attention_scores = jnp.where(attention_mask, attention_scores, -MAX_EXP)

            attention_weights = jnp.exp(attention_scores - log_sum_exp_block)
            # Zero out weights for masked key-query pairs BEFORE computing any gradients.
            attention_weights = jnp.where(attention_mask, attention_weights, 0.0)

            # dV
            value_grad = jnp.matmul(attention_weights.swapaxes(-1, -2), grad_input_block)
            value_grad_existing = lax.dynamic_slice_in_dim(grad_value, kv_start_idx, kv_block_size, axis=1)
            grad_value = lax.dynamic_update_slice_in_dim(
                grad_value, value_grad_existing + value_grad, kv_start_idx, axis=1)

            # dP -> dS  (scale folded into dS to avoid extra multiply in dQ/dK)
            dP_ij = jnp.matmul(grad_input_block, value_block.swapaxes(-1, -2))
            dS = attention_weights * (dP_ij - output_dot_grad_block) * scale

            # dQ
            grad_query_block = grad_query_block + jnp.matmul(dS, key_block)

            # dK
            key_grad = jnp.matmul(dS.swapaxes(-1, -2), query_block)
            key_grad_existing = lax.dynamic_slice_in_dim(grad_key, kv_start_idx, kv_block_size, axis=1)
            grad_key = lax.dynamic_update_slice_in_dim(
                grad_key, key_grad_existing + key_grad, kv_start_idx, axis=1)

            return (grad_query_block, grad_key, grad_value)

        inner_carry_init = (grad_query_block, grad_key, grad_value)
        grad_query_block, grad_key, grad_value = \
            lax.fori_loop(0, num_kv_blocks, kv_block_loop, inner_carry_init)

        # Zero gradients for fully-masked query positions.
        grad_query_block = jnp.where(has_valid_block[..., None], grad_query_block, 0.0)

        query_grad_existing = lax.dynamic_slice_in_dim(grad_query, query_start_idx, query_block_size, axis=1)
        grad_query = lax.dynamic_update_slice_in_dim(
            grad_query, query_grad_existing + grad_query_block, query_start_idx, axis=1)

        return (grad_query, grad_key, grad_value)

    init_carry = (grad_query, grad_key, grad_value)
    grad_query, grad_key, grad_value = \
        lax.fori_loop(0, num_query_blocks, query_block_loop, init_carry)

    # Return gradients for q, k, v, padding_mask_k, padding_mask_q, causal
    # (last three are None — non-differentiable).
    return (grad_query, grad_key, grad_value, None, None, None)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def flash_attention(q, k, v, padding_mask_k=None, padding_mask_q=None,
                    causal=False, block_size_q=128, block_size_kv=128):
    """Flash Attention with custom VJP.

    Args:
        q:              [batch_heads, query_seq_len, head_dim]
        k:              [batch_heads, key_seq_len, head_dim]
        v:              [batch_heads, key_seq_len, value_dim]
        padding_mask_k: [batch_heads, key_seq_len]   (True = keep)
        padding_mask_q: [batch_heads, query_seq_len] (True = keep)
        causal:         apply causal (lower-triangular) mask
        block_size_q:   tile size along query dimension
        block_size_kv:  tile size along key/value dimension

    Returns:
        output: [batch_heads, query_seq_len, value_dim]
    """
    output, _ = flash_attention_forward(
        q, k, v, padding_mask_k, padding_mask_q, causal, block_size_q, block_size_kv)
    return output


flash_attention.defvjp(flash_attention_forward, flash_attention_backward)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlashAttention(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)

        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 64
        self.head_dim = 64

        q_key, k_key, v_key, mask_key = jax.random.split(self.key, 4)
        bh = self.batch_size * self.num_heads
        self.q = jax.random.normal(q_key, (bh, self.seq_len, self.head_dim))
        self.k = jax.random.normal(k_key, (bh, self.seq_len, self.head_dim))
        self.v = jax.random.normal(v_key, (bh, self.seq_len, self.head_dim))

        self.padding_mask = jax.random.bernoulli(mask_key, 0.1, (self.batch_size, self.seq_len))
        self.padding_mask = jnp.repeat(self.padding_mask, self.num_heads, axis=0)

        self.flash_attention_jit = jax.jit(partial(
            flash_attention, block_size_q=32, block_size_kv=32))

    def standard_attention(self, q, k, v, padding_mask_k=None, padding_mask_q=None, causal=False):
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(k.shape[-1])

        if causal:
            scores = jnp.where(jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool)), scores, -jnp.inf)
        if padding_mask_k is not None:
            scores = jnp.where(padding_mask_k[:, None, :], scores, -jnp.inf)
        if padding_mask_q is not None:
            scores = jnp.where(padding_mask_q[:, :, None], scores, -jnp.inf)

        row_valid = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
        scores = jnp.where(row_valid, scores, 0.0)

        attn_weights = jax.nn.softmax(scores)
        output = jnp.matmul(attn_weights, v)
        # Zero out fully masked rows to match flash attention convention.
        output = jnp.where(row_valid, output, 0.0)
        return output

    def test_forward_no_mask(self):
        flash_output = self.flash_attention_jit(self.q, self.k, self.v, None, None, False)
        standard_output = self.standard_attention(self.q, self.k, self.v, None, None, False)
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"无掩码前向差异: {diff:.6f}")
        self.assertLess(diff, 1e-3)

    def test_forward_with_mask(self):
        flash_output = self.flash_attention_jit(self.q, self.k, self.v,
                                                 self.padding_mask, self.padding_mask, False)
        standard_output = self.standard_attention(self.q, self.k, self.v,
                                                   self.padding_mask, self.padding_mask, False)
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"带掩码前向差异: {diff:.6f}")
        self.assertLess(diff, 1e-3)

    def test_forward_causal_mask(self):
        flash_output = self.flash_attention_jit(self.q, self.k, self.v, None, None, True)
        standard_output = self.standard_attention(self.q, self.k, self.v, None, None, True)
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"因果掩码前向差异: {diff:.6f}")
        self.assertLess(diff, 1e-3)

    def test_backward_no_mask(self):
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_jit(q, k, v, None, None, False))
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(q, k, v, None, None, False))

        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

        dq = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        dk = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        dv = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        print(f"无掩码梯度差异 - Q: {dq:.6f}, K: {dk:.6f}, V: {dv:.6f}")
        self.assertLess(dq, 1e-3)
        self.assertLess(dk, 1e-3)
        self.assertLess(dv, 1e-3)

    def test_backward_with_mask(self):
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_jit(
                q, k, v, self.padding_mask, self.padding_mask, False))
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(
                q, k, v, self.padding_mask, self.padding_mask, False))

        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

        dq = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        dk = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        dv = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        print(f"带掩码梯度差异 - Q: {dq:.6f}, K: {dk:.6f}, V: {dv:.6f}")
        self.assertLess(dq, 1e-3)
        self.assertLess(dk, 1e-3)
        self.assertLess(dv, 1e-3)

    def test_backward_causal_mask(self):
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_jit(q, k, v, None, None, True))
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(q, k, v, None, None, True))

        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

        dq = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        dk = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        dv = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        print(f"因果掩码梯度差异 - Q: {dq:.6f}, K: {dk:.6f}, V: {dv:.6f}")
        # epsilon bug 已修复（_MIN_NORMALIZER=1e-12），所有块大小数值一致。
        self.assertLess(dq, 1e-3)
        self.assertLess(dk, 1e-3)
        self.assertLess(dv, 1e-3)

    def test_numerical_stability(self):
        q = jnp.full_like(self.q, 1000.0)
        k = jnp.full_like(self.k, -1000.0)
        v = jnp.full_like(self.v, 1e6)
        output = self.flash_attention_jit(q, k, v, self.padding_mask, self.padding_mask, False)
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isinf(output)))
        print("数值稳定性测试通过")

    def test_different_block_sizes(self):
        block_sizes = [(16, 16), (32, 32), (64, 64), (16, 32), (32, 16)]
        outputs = []
        for bs_q, bs_kv in block_sizes:
            output = flash_attention(self.q, self.k, self.v,
                                     self.padding_mask, self.padding_mask,
                                     False, bs_q, bs_kv)
            outputs.append(output)
        for i in range(1, len(outputs)):
            diff = jnp.max(jnp.abs(outputs[0] - outputs[i]))
            self.assertLess(diff, 1e-3,
                            f"块大小 {block_sizes[0]} 与 {block_sizes[i]} 输出不一致")
        print("不同块大小一致性测试通过")

    def test_all_masked_rows(self):
        """全 mask 行应输出零，而非数值垃圾。"""
        mask = jnp.ones((self.batch_size * self.num_heads, self.seq_len), dtype=bool)
        mask = mask.at[:, :32].set(False)

        output = self.flash_attention_jit(self.q, self.k, self.v, mask, mask, False)
        masked_slice = output[:, :32, :]
        self.assertLess(jnp.max(jnp.abs(masked_slice)), 1e-6,
                        "全 mask 行输出应为零")
        print("全 mask 行测试通过")


if __name__ == "__main__":
    unittest.main()
