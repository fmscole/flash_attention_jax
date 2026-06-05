# Flash Attention JAX

JAX implementation of Flash Attention (v1 & v2) with custom VJPs for memory-efficient exact attention. Built on top are Transformer building blocks, a Vision Transformer (ViT) for CIFAR-10, and an English→Chinese translation model.

## Installation

```bash
pip install jax optax torch torchvision numpy matplotlib jieba opencc
```

No package install needed — import from `src/` directly:

```python
import sys
sys.path.insert(0, 'src')
from flash_attention import flash_attention, flash_attention_v2
```

## Flash Attention

The core module provides two variants of block-sparse Flash Attention, both with manual VJP (custom gradient) implementations that avoid materializing the full `seq_len × seq_len` attention matrix.

### API

Both functions share the same signature:

```
flash_attention(q, k, v, padding_mask_k=None, padding_mask_q=None,
                causal=False, block_size_q=128, block_size_kv=128)
flash_attention_v2(q, k, v, padding_mask_k=None, padding_mask_q=None,
                   causal=False, block_size_q=128, block_size_kv=128)
```

**Inputs:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `q` | `[batch_heads, seq_len, head_dim]` | Query. `batch_heads = batch_size × num_heads` |
| `k` | `[batch_heads, kv_seq_len, head_dim]` | Key |
| `v` | `[batch_heads, kv_seq_len, head_dim]` | Value |
| `padding_mask_k` | `[batch_heads, kv_seq_len]` or `None` | Boolean mask. `True` = valid, `False` = pad |
| `padding_mask_q` | `[batch_heads, seq_len]` or `None` | Same convention as above |
| `causal` | `bool` | If `True`, applies causal masking |
| `block_size_q` | `int` | Tile size along query dimension |
| `block_size_kv` | `int` | Tile size along key/value dimension |

**Returns:** `[batch_heads, seq_len, head_dim]`

**Important:** Inputs use `[batch_heads, ...]` format (batch and heads merged into one dimension). Use `q.reshape(batch * n_heads, seq, d)` before calling.

### v1 vs v2

| | v1 (`flash_attention`) | v2 (`flash_attention_v2`) |
|---|---|---|
| **Forward loop** | Q-block outer, KV-block inner | KV-block outer, Q-block inner |
| **Backward loop** | Same as forward (Q-outer, KV-inner) | Same as forward (KV-outer, Q-inner) |
| **Reference** | Standard Flash Attention | FlashAttention-2 (Dao 2023) §3.2 |
| **Numerical** | Identical to v2 (diff < 1e-5) | Identical to v1 |

v2's KV-outer backward loop reduces SRAM writes compared to v1, matching the FlashAttention-2 paper's sequence-length parallelism optimization. Both produce identical forward/gradient values.

```python
from flash_attention import flash_attention, flash_attention_v2

# Same call for both
out1 = flash_attention(q, k, v, causal=True, block_size_q=128, block_size_kv=128)
out2 = flash_attention_v2(q, k, v, causal=True, block_size_q=128, block_size_kv=128)
assert jnp.allclose(out1, out2, atol=1e-5)
```

### Numerical Stability

When `block_size < seq_len`, the attention is computed in tiles. The normalization uses a unified epsilon (`1e-12`) shared between the forward softmax and the backward log-sum-exp recovery, ensuring the backward `exp(S - L)` exactly recovers the forward weights. This eliminates gradient drift that can cause training collapse at large sequence lengths.

### Running Tests

```bash
# Unit tests for both v1 and v2 (self-contained, no extra deps)
python src/flash_attention/flash_attention_v1.py
python src/flash_attention/flash_attention_v2.py

# Dedicated v1 test suite
python src/flash_attention/flash_attention_test.py
```

Each file includes a `__main__` block with test classes covering forward/backward with and without masks, causal mode, numerical stability, and block-size variants.

## Transformers

Building blocks on top of Flash Attention. Two parallel implementations: `transformer_flash_v1` (uses `flash_attention`) and `transformer_flash_v2` (uses `flash_attention_v2`). They have identical APIs and interchangeable outputs.

### Components

```python
from transformers.transformer_flash_v1 import (
    Embedding, PositionalEncoding,
    MultiHeadSelfAttention, MultiHeadCrossAttention,
    TransformerEncoderBlock, TransformerDecoderBlock,
    TransformerEncoder, TransformerDecoder,
    Transformer,  # full encoder-decoder model
)
```

| Component | Description |
|-----------|-------------|
| `Embedding(num_embeddings, dim)` | Token embedding via `jnp.take` |
| `PositionalEncoding(max_len, dim)` | Learned positional encoding |
| `MultiHeadSelfAttention(n_heads, head_dim, causal, block_size_q, block_size_kv)` | QKV projection → flash attention → output projection |
| `MultiHeadCrossAttention(n_heads, head_dim, ...)` | Cross-attention with separate Q from K/V |
| `TransformerEncoderBlock(n_heads, head_dim, embed_dim, mlp_dim, dropout_rate)` | Pre-LN: Self-Attn → FFN, both with residual |
| `TransformerDecoderBlock(n_heads, head_dim, embed_dim, mlp_dim, dropout_rate)` | Pre-LN: Causal Self-Attn → Cross-Attn → FFN |
| `TransformerEncoder(num_layers, ...)` | Stacks encoder blocks + final LayerNorm |
| `TransformerDecoder(num_layers, ...)` | Stacks decoder blocks + final LayerNorm |
| `Transformer(src_vocab_size, tgt_vocab_size, embed_dim, n_heads, head_dim, mlp_dim, num_encoder_layers, num_decoder_layers, max_len, block_size)` | Full encoder-decoder. Returns `(init, apply, encode, decode)` |

All Transformer modules use **Pre-LN** architecture. Dropout is controlled via the `is_training` keyword argument to `apply`.

### Shape Convention

The Transformer modules flatten batch and head dimensions internally. Users work with standard shapes:

```python
# Input: [batch, seq_len]
# The Transformer handles the [batch, seq_len] → [batch_heads, seq, head_dim] reshaping internally.
```

### MAE Utilities

```python
from transformers.mae_utils import random_masking, unshuffle
```

| Function | Purpose |
|----------|---------|
| `random_masking(x, mask_ratio, rng)` → `(x_visible, ids_restore, mask)` | Randomly masks patches for MAE pretraining |
| `unshuffle(x_visible, ids_restore, mask_token, num_patches)` | Restores encoded visible patches to full sequence with mask tokens |

### Layer Library (`lib.stax_plus`)

Extended JAX stax layer library used internally by the Transformer modules:

```python
from lib.stax_plus import Dense, Conv, LayerNorm, BatchNorm, Dropout, Gelu, serial, parallel
```

Stateful-aware `serial()`/`parallel()` combinators transparently thread BatchNorm state and handle 3-tuple init returns.

## ViT: Vision Transformer (CIFAR-10)

A Vision Transformer for CIFAR-10 classification.

```
src/vit/
└── vit_cifar10_stax.py              # End-to-end supervised training
```

### Usage

```bash
python src/vit/vit_cifar10_stax.py
```

### Architecture

| Component | Config |
|-----------|--------|
| Patch embed | Conv 4×4, stride 4 → 64 patches |
| Embed dim | 384 |
| Heads | 6 |
| Layers | 7 |
| MLP dim | 768 |
| Positional encoding | Learned (not sinusoidal) |
| Attention | Flash Attention v1 |

### Training Details

| Config | Value |
|--------|-------|
| Epochs | 500 |
| Peak LR | 3e-4, warmup 10 epochs + cosine decay |
| Batch size | 128 |
| Optimizer | AdamW (weight_decay=0.1) |
| Data augmentation | RandAugment + MixUp(α=0.8) |
| Label smoothing | 0.1 |
| Accuracy | 90~91% |

## Translation: EN→ZH

Encoder-decoder Transformer for English→Chinese translation on the AiChallenger dataset (~10M sentence pairs).

```
src/translation/
├── prepare_ai_challenger.py              # Dataset preprocessing
└── translate_stax_flash.py               # Single-stage end-to-end training
```

### Dataset Preparation

```bash
# Generate training TSV from raw AiChallenger files
python src/translation/prepare_ai_challenger.py
```

Requires the [AiChallenger Machine Translation dataset](https://aistudio.baidu.com/datasetdetail/220848).

### Usage

```bash
# Train from scratch on parallel data
python src/translation/translate_stax_flash.py
```

### Architecture

| Component | Config |
|-----------|--------|
| Encoder / Decoder | 6 layers each, Pre-LN |
| Embed dim | 512 (8 heads × 64) |
| FFN dim | 2048 |
| Vocab | ~55K (frequency-based, BPE-free) |
| Max length | 64 |
| Attention | Flash Attention v1 |
| Inference | Greedy decoding |

### Training Details

| Config | Value |
|--------|-------|
| Epochs | 10000 (early stop) |
| Peak LR | 3e-5, warmup 4000 steps, cosine decay → 1e-5 |
| Batch size | 64 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Grad clip | global norm 1.0 |
| Early stop | val loss no improvement for 3 validations |
| Inference | Greedy decoding, stop at `<eos>` |

## License

MIT
