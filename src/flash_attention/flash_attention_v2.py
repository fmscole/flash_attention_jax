"""FlashAttention-2 with custom VJP for JAX.

Reference: Dao, "FlashAttention-2: Faster Attention with Better Parallelism
and Work Partitioning", 2023.

Key changes from FlashAttention-1 (v1):
  - Forward:  unscaled accumulator Õ til the final Q-block, then O = Õ / ℓ.
             Only logsumexp L = m + log(ℓ) is saved (not m, ℓ separately).
  - Backward: loop order reversed — outer loop over KV blocks, inner over Q
             blocks.  This maps to the "sequence-length parallelism" described
             in FA2 §3.2 and the warp-partitioning in §3.3.

Input shapes:
    q, k, v: [batch_heads, seq_len, head_dim]  (batch_heads = batch_size * num_heads)
    padding_mask_k, padding_mask_q: [batch_heads, seq_len] (True = keep)
"""

import unittest
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

MAX_EXP = 50.0

# Numerical-stability fix (same as v1):
# Forward normalisation and log-sum-exp saving must use the same epsilon so
# that exp(S - L) in the backward recovers the exact forward attention weights.
_MIN_NORMALIZER = 1e-12


def _make_attention_mask(causal, query_start_idx, query_block_size,
                          kv_start_idx, kv_block_size,
                          query_mask_block, key_mask_block):
    """Combined attention mask for a Q-K block pair (shared by fwd/bwd)."""

    def _causal_branch():
        query_indices = jnp.arange(query_block_size) + query_start_idx
        key_indices = jnp.arange(kv_block_size) + kv_start_idx
        causal_mask = (query_indices[:, None] >= key_indices[None, :])
        causal_mask = causal_mask[None, :, :]
        return causal_mask & query_mask_block[:, :, None] & key_mask_block[:, None, :]

    def _padding_branch():
        return query_mask_block[:, :, None] & key_mask_block[:, None, :]

    return lax.cond(causal, _causal_branch, _padding_branch)


# ═══════════════════════════════════════════════════════════════════════════════
# Forward pass  —  Algorithm 1 of FlashAttention-2
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=['block_size_q', 'block_size_kv'])
def flash_attention_v2_forward(query, key, value, key_padding_mask,
                               query_padding_mask, causal, block_size_q, block_size_kv):
    """FlashAttention-2 forward pass (Algorithm 1).

    Uses an *un-scaled* running output Õ through all KV blocks; divides by ℓ
    only at the end.  Saves only logsumexp L (not m, ℓ separately).
    """
    batch_heads, query_seq_len, head_dim = query.shape
    _, key_seq_len, value_dim = value.shape

    # Mask defaults
    if key_padding_mask is None:
        key_padding_mask = jnp.ones((batch_heads, key_seq_len), dtype=bool)
    if query_padding_mask is None:
        query_padding_mask = jnp.ones((batch_heads, query_seq_len), dtype=bool)

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

    # Output: unscaled accumulator Õ final → divided by ℓ at Q-block exit.
    # Use value_dim for correctness under multi-query attention (head_dim ≠ value_dim).
    output = jnp.zeros((batch_heads, query_seq_len, value_dim), dtype=query.dtype)
    L = jnp.full((batch_heads, query_seq_len), -jnp.inf, dtype=jnp.float32)
    has_valid_global = jnp.zeros((batch_heads, query_seq_len), dtype=bool)

    # ---- KV-block loop (inner) ----
    def kv_block_loop(kv_block_idx, carry):
        O_tilde, ell, m, block_query, query_start_idx, query_mask_block, has_valid = carry
        kv_start_idx = kv_block_idx * kv_block_size

        key_block = lax.dynamic_slice_in_dim(key, kv_start_idx, kv_block_size, axis=1)
        value_block = lax.dynamic_slice_in_dim(value, kv_start_idx, kv_block_size, axis=1)
        key_mask_block = lax.dynamic_slice_in_dim(key_padding_mask, kv_start_idx, kv_block_size, axis=1)

        attention_mask = _make_attention_mask(
            causal, query_start_idx, query_block_size,
            kv_start_idx, kv_block_size,
            query_mask_block, key_mask_block)

        # S = Q_i K_j^T / sqrt(d)
        S = jnp.matmul(block_query, key_block.swapaxes(-1, -2)) * scale
        S = jnp.clip(S, -MAX_EXP, MAX_EXP)
        S = jnp.where(attention_mask, S, -MAX_EXP)

        has_valid = has_valid | jnp.any(attention_mask, axis=-1)

        # Online softmax update (Algorithm 1 lines 9-10)
        m_new = jnp.maximum(m[..., None], jnp.max(S, axis=-1, keepdims=True))          # (9a)
        P_tilde = jnp.exp(S - m_new)                                                     # (9b)
        ell_new = (jnp.exp(m[..., None] - m_new) * ell[..., None] +
                   jnp.sum(P_tilde, axis=-1, keepdims=True))                            # (9c)
        O_tilde = (jnp.exp(m[..., None] - m_new) * O_tilde +
                   jnp.matmul(P_tilde, value_block))                                     # (10)

        m_new = jnp.squeeze(m_new, axis=-1)
        ell_new = jnp.squeeze(ell_new, axis=-1)

        return (O_tilde, ell_new, m_new, block_query, query_start_idx, query_mask_block, has_valid)

    # ---- Q-block loop (outer) ----
    def query_block_loop(query_block_idx, state):
        output, L, has_valid_global = state
        query_start_idx = query_block_idx * query_block_size

        query_block = lax.dynamic_slice_in_dim(query, query_start_idx, query_block_size, axis=1)
        query_mask_block = lax.dynamic_slice_in_dim(query_padding_mask, query_start_idx,
                                                     query_block_size, axis=1)

        O_tilde = jnp.zeros((batch_heads, query_block_size, value_dim), dtype=query.dtype)
        ell = jnp.zeros((batch_heads, query_block_size), dtype=jnp.float32)
        m = jnp.full((batch_heads, query_block_size), -jnp.inf, dtype=jnp.float32)
        has_valid_block = jnp.zeros((batch_heads, query_block_size), dtype=bool)

        carry_init = (O_tilde, ell, m, query_block, query_start_idx, query_mask_block, has_valid_block)
        O_tilde, ell, m, _, _, _, has_valid_block = \
            lax.fori_loop(0, num_kv_blocks, kv_block_loop, carry_init)

        # Algorithm 1 line 12: O_i = Õ_i / ℓ_i
        O_i = O_tilde / jnp.maximum(ell[..., None], _MIN_NORMALIZER)
        # Algorithm 1 line 13: L_i = m_i + log(ℓ_i)
        L_i = jnp.where(jnp.isinf(m),
                        m,
                        m + jnp.log(jnp.maximum(ell, _MIN_NORMALIZER)))

        output = lax.dynamic_update_slice_in_dim(output, O_i, query_start_idx, axis=1)
        L = lax.dynamic_update_slice_in_dim(L, L_i, query_start_idx, axis=1)
        has_valid_global = lax.dynamic_update_slice_in_dim(has_valid_global, has_valid_block,
                                                           query_start_idx, axis=1)
        return (output, L, has_valid_global)

    init_state = (output, L, has_valid_global)
    output, L, has_valid_global = \
        lax.fori_loop(0, num_query_blocks, query_block_loop, init_state)

    # Zero out fully-masked query positions.
    output = jnp.where(has_valid_global[..., None], output, 0.0)

    residuals = (query, key, value, L, output, causal,
                 key_padding_mask, query_padding_mask, has_valid_global)
    return output, residuals


# ═══════════════════════════════════════════════════════════════════════════════
# Backward pass  —  Algorithm 2 of FlashAttention-2
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key difference from v1: outer loop over KV blocks (j), inner loop over Q
# blocks (i).  This maps to §3.2 parallelism — each thread block handles one
# column block, and atomic adds are used for dQ updates.  In JAX immutability
# we use dynamic_update_slice_in_dim instead of atomics.

@partial(jax.jit, static_argnames=['block_size_q', 'block_size_kv'])
def flash_attention_v2_backward(block_size_q, block_size_kv, residuals, grad_input):
    """FlashAttention-2 backward pass (Algorithm 2).

    Loop order: KV-outer, Q-inner (reversed from v1).
    """
    query, key, value, L, output, causal, \
        key_padding_mask, query_padding_mask, has_valid_global = residuals

    batch_heads, query_seq_len, head_dim = query.shape
    _, key_seq_len, value_dim = value.shape

    if key_padding_mask is None:
        key_padding_mask = jnp.ones((batch_heads, key_seq_len), dtype=bool)
    if query_padding_mask is None:
        query_padding_mask = jnp.ones((batch_heads, query_seq_len), dtype=bool)

    scale = 1.0 / jnp.sqrt(head_dim)

    # Algorithm 2 line 4: D = rowsum(dO ∘ O)
    D = jnp.sum(output * grad_input, axis=-1)  # [batch_heads, query_seq_len]

    query_block_size = min(block_size_q, query_seq_len)
    kv_block_size = min(block_size_kv, key_seq_len)
    num_query_blocks = query_seq_len // query_block_size
    num_kv_blocks = key_seq_len // kv_block_size

    grad_query = jnp.zeros_like(query)
    grad_key = jnp.zeros_like(key)
    grad_value = jnp.zeros_like(value)

    # ---- Outer loop: KV blocks (Algorithm 2 line 5) ----
    def kv_block_loop(kv_block_idx, carry):
        grad_query, grad_key, grad_value = carry
        kv_start_idx = kv_block_idx * kv_block_size

        # Algorithm 2 line 6: load K_j, V_j
        key_block = lax.dynamic_slice_in_dim(key, kv_start_idx, kv_block_size, axis=1)
        value_block = lax.dynamic_slice_in_dim(value, kv_start_idx, kv_block_size, axis=1)
        key_mask_block = lax.dynamic_slice_in_dim(key_padding_mask, kv_start_idx, kv_block_size, axis=1)

        # Algorithm 2 line 7: initialize dK_j, dV_j
        dK_j = jnp.zeros((batch_heads, kv_block_size, head_dim), dtype=key.dtype)
        dV_j = jnp.zeros((batch_heads, kv_block_size, value_dim), dtype=value.dtype)

        # ---- Inner loop: Q blocks (Algorithm 2 line 8) ----
        def query_block_loop(query_block_idx, inner_carry):
            dK_j, dV_j, grad_query = inner_carry
            query_start_idx = query_block_idx * query_block_size

            # Algorithm 2 line 9: load Q_i, O_i, dO_i, dQ_i, L_i, D_i
            query_block = lax.dynamic_slice_in_dim(query, query_start_idx, query_block_size, axis=1)
            grad_input_block = lax.dynamic_slice_in_dim(grad_input, query_start_idx, query_block_size, axis=1)
            L_block = lax.dynamic_slice_in_dim(L, query_start_idx, query_block_size, axis=1)
            D_block = lax.dynamic_slice_in_dim(D, query_start_idx, query_block_size, axis=1)
            has_valid_block = lax.dynamic_slice_in_dim(has_valid_global, query_start_idx, query_block_size, axis=1)
            query_mask_block = lax.dynamic_slice_in_dim(query_padding_mask, query_start_idx, query_block_size, axis=1)

            attention_mask = _make_attention_mask(
                causal, query_start_idx, query_block_size,
                kv_start_idx, kv_block_size,
                query_mask_block, key_mask_block)

            # Algorithm 2 line 10: S = Q_i K_j^T
            S = jnp.matmul(query_block, key_block.swapaxes(-1, -2)) * scale
            S = jnp.clip(S, -MAX_EXP, MAX_EXP)
            S = jnp.where(attention_mask, S, -MAX_EXP)

            # Algorithm 2 line 11: P = exp(S - L)
            L_block_exp = L_block[..., None]
            P = jnp.exp(S - L_block_exp)
            P = jnp.where(attention_mask, P, 0.0)

            # Algorithm 2 line 12: dV_j += P^T dO_i
            dV_j = dV_j + jnp.matmul(P.swapaxes(-1, -2), grad_input_block)

            # Algorithm 2 line 13: dP = dO_i V_j^T
            dP = jnp.matmul(grad_input_block, value_block.swapaxes(-1, -2))

            # Algorithm 2 line 14: dS = P ∘ (dP - D)   (scale folded in)
            D_block_exp = D_block[..., None]
            dS = P * (dP - D_block_exp) * scale

            # Algorithm 2 line 15: dQ_i ← dQ_i + dS K_j   (atomic in CUDA)
            dQ_i = jnp.matmul(dS, key_block)
            dQ_i = jnp.where(has_valid_block[..., None], dQ_i, 0.0)
            dQ_existing = lax.dynamic_slice_in_dim(grad_query, query_start_idx, query_block_size, axis=1)
            grad_query = lax.dynamic_update_slice_in_dim(
                grad_query, dQ_existing + dQ_i, query_start_idx, axis=1)

            # Algorithm 2 line 16: dK_j += dS^T Q_i
            dK_j = dK_j + jnp.matmul(dS.swapaxes(-1, -2), query_block)

            return (dK_j, dV_j, grad_query)

        inner_carry_init = (dK_j, dV_j, grad_query)
        dK_j, dV_j, grad_query = \
            lax.fori_loop(0, num_query_blocks, query_block_loop, inner_carry_init)

        # Algorithm 2 line 18: write dK_j, dV_j to HBM
        dK_existing = lax.dynamic_slice_in_dim(grad_key, kv_start_idx, kv_block_size, axis=1)
        grad_key = lax.dynamic_update_slice_in_dim(
            grad_key, dK_existing + dK_j, kv_start_idx, axis=1)
        dV_existing = lax.dynamic_slice_in_dim(grad_value, kv_start_idx, kv_block_size, axis=1)
        grad_value = lax.dynamic_update_slice_in_dim(
            grad_value, dV_existing + dV_j, kv_start_idx, axis=1)

        return (grad_query, grad_key, grad_value)

    init_carry = (grad_query, grad_key, grad_value)
    grad_query, grad_key, grad_value = \
        lax.fori_loop(0, num_kv_blocks, kv_block_loop, init_carry)

    # Return grads for q, k, v; None for non-differentiable args.
    return (grad_query, grad_key, grad_value, None, None, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def flash_attention_v2(q, k, v, padding_mask_k=None, padding_mask_q=None,
                       causal=False, block_size_q=128, block_size_kv=128):
    """FlashAttention-2 with custom VJP.

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
    output, _ = flash_attention_v2_forward(
        q, k, v, padding_mask_k, padding_mask_q, causal, block_size_q, block_size_kv)
    return output


flash_attention_v2.defvjp(flash_attention_v2_forward, flash_attention_v2_backward)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlashAttentionV2(unittest.TestCase):
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

        self.flash_attention_v2_jit = jax.jit(partial(
            flash_attention_v2, block_size_q=32, block_size_kv=32))

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
        output = jnp.where(row_valid, output, 0.0)
        return output

    def test_forward_no_mask(self):
        flash_output = self.flash_attention_v2_jit(self.q, self.k, self.v, None, None, False)
        standard_output = self.standard_attention(self.q, self.k, self.v, None, None, False)
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"FA2 无掩码前向差异: {diff:.6f}")
        self.assertLess(diff, 1e-3)

    def test_forward_with_mask(self):
        flash_output = self.flash_attention_v2_jit(self.q, self.k, self.v,
                                                    self.padding_mask, self.padding_mask, False)
        standard_output = self.standard_attention(self.q, self.k, self.v,
                                                   self.padding_mask, self.padding_mask, False)
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"FA2 带掩码前向差异: {diff:.6f}")
        self.assertLess(diff, 1e-3)

    def test_forward_causal_mask(self):
        flash_output = self.flash_attention_v2_jit(self.q, self.k, self.v, None, None, True)
        standard_output = self.standard_attention(self.q, self.k, self.v, None, None, True)
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"FA2 因果掩码前向差异: {diff:.6f}")
        self.assertLess(diff, 1e-3)

    def test_backward_no_mask(self):
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_v2_jit(q, k, v, None, None, False))
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(q, k, v, None, None, False))

        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

        dq = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        dk = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        dv = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        print(f"FA2 无掩码梯度差异 - Q: {dq:.6f}, K: {dk:.6f}, V: {dv:.6f}")
        self.assertLess(dq, 1e-3)
        self.assertLess(dk, 1e-3)
        self.assertLess(dv, 1e-3)

    def test_backward_with_mask(self):
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_v2_jit(
                q, k, v, self.padding_mask, self.padding_mask, False))
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(
                q, k, v, self.padding_mask, self.padding_mask, False))

        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

        dq = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        dk = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        dv = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        print(f"FA2 带掩码梯度差异 - Q: {dq:.6f}, K: {dk:.6f}, V: {dv:.6f}")
        self.assertLess(dq, 1e-3)
        self.assertLess(dk, 1e-3)
        self.assertLess(dv, 1e-3)

    def test_backward_causal_mask(self):
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_v2_jit(q, k, v, None, None, True))
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(q, k, v, None, None, True))

        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)

        dq = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        dk = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        dv = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        print(f"FA2 因果掩码梯度差异 - Q: {dq:.6f}, K: {dk:.6f}, V: {dv:.6f}")
        # Causal backward tolerates 2e-3 — tiled softmax order differs from standard,
        # and the causal mask amplifies accumulation differences slightly.
        self.assertLess(dq, 2e-3)
        self.assertLess(dk, 2e-3)
        self.assertLess(dv, 2e-3)

    def test_v1_v2_consistency(self):
        """FA2 应与 FA1 输出一致（随机输入下）。"""
        try:
            from flash_attention_v1 import flash_attention as flash_attention_v1
        except ModuleNotFoundError:
            from src.flash_attention.flash_attention_v1 import flash_attention as flash_attention_v1

        v1_out = flash_attention_v1(self.q, self.k, self.v,
                                     self.padding_mask, self.padding_mask,
                                     False, 32, 32)
        v2_out = flash_attention_v2(self.q, self.k, self.v,
                                     self.padding_mask, self.padding_mask,
                                     False, 32, 32)
        diff = jnp.max(jnp.abs(v1_out - v2_out))
        print(f"V1 vs V2 前向差异: {diff:.6f}")
        self.assertLess(diff, 1e-5)

        # Gradient consistency
        def loss_v1(q, k, v):
            return jnp.sum(flash_attention_v1(q, k, v, self.padding_mask, self.padding_mask, False, 32, 32))
        def loss_v2(q, k, v):
            return jnp.sum(flash_attention_v2(q, k, v, self.padding_mask, self.padding_mask, False, 32, 32))

        gv1 = jax.grad(loss_v1, (0, 1, 2))(self.q, self.k, self.v)
        gv2 = jax.grad(loss_v2, (0, 1, 2))(self.q, self.k, self.v)

        dq = jnp.max(jnp.abs(gv1[0] - gv2[0]))
        dk = jnp.max(jnp.abs(gv1[1] - gv2[1]))
        dv = jnp.max(jnp.abs(gv1[2] - gv2[2]))
        print(f"V1 vs V2 梯度差异 - Q: {dq:.6f}, K: {dk:.6f}, V: {dv:.6f}")
        self.assertLess(dq, 1e-5)
        self.assertLess(dk, 1e-5)
        self.assertLess(dv, 1e-5)

    def test_numerical_stability(self):
        q = jnp.full_like(self.q, 1000.0)
        k = jnp.full_like(self.k, -1000.0)
        v = jnp.full_like(self.v, 1e6)
        output = self.flash_attention_v2_jit(q, k, v, self.padding_mask, self.padding_mask, False)
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isinf(output)))
        print("FA2 数值稳定性测试通过")

    def test_different_block_sizes(self):
        block_sizes = [(16, 16), (32, 32), (64, 64), (16, 32), (32, 16)]
        outputs = []
        for bs_q, bs_kv in block_sizes:
            output = flash_attention_v2(self.q, self.k, self.v,
                                         self.padding_mask, self.padding_mask,
                                         False, bs_q, bs_kv)
            outputs.append(output)
        for i in range(1, len(outputs)):
            diff = jnp.max(jnp.abs(outputs[0] - outputs[i]))
            self.assertLess(diff, 1e-3,
                            f"块大小 {block_sizes[0]} 与 {block_sizes[i]} 输出不一致")
        print("FA2 不同块大小一致性测试通过")

    def test_all_masked_rows(self):
        mask = jnp.ones((self.batch_size * self.num_heads, self.seq_len), dtype=bool)
        mask = mask.at[:, :32].set(False)
        output = self.flash_attention_v2_jit(self.q, self.k, self.v, mask, mask, False)
        masked_slice = output[:, :32, :]
        self.assertLess(jnp.max(jnp.abs(masked_slice)), 1e-6,
                        "全 mask 行输出应为零")
        print("FA2 全 mask 行测试通过")


if __name__ == "__main__":
    unittest.main()
