"""
MAE (Masked Autoencoder) utility functions for JAX/Stax ViT.

Provides the core masking / unshuffling operations needed for
joint supervised + self-supervised training of Vision Transformers.

Reference: He et al. "Masked Autoencoders Are Scalable Vision Learners" (2022)
"""

import jax
import jax.numpy as jnp


def random_masking(x, mask_ratio, rng):
    """Randomly mask patches for MAE pre-training.

    Performs random shuffling and keeps the first ``len_keep`` patches.
    Returns the visible patches, inverse permutation indices, and a boolean mask.

    Args:
        x: (batch, num_patches, dim) — input patch embeddings
        mask_ratio: float in [0, 1) — fraction of patches to mask
        rng: JAX PRNG key

    Returns:
        x_visible:   (batch, len_keep, dim) — unmasked (visible) patches
        ids_restore: (batch, num_patches)   — inverse permutation; maps
                      shuffled position → original position
        mask:        (batch, num_patches)   — bool, True = masked position
    """
    batch, num_patches, dim = x.shape
    len_keep = int(num_patches * (1.0 - mask_ratio))

    # Generate random noise per patch, sort → random permutation
    noise = jax.random.uniform(rng, (batch, num_patches))
    ids_shuffle = jnp.argsort(noise, axis=1)          # ascending: small noise → keep
    ids_restore = jnp.argsort(ids_shuffle, axis=1)    # inverse permutation
    ids_keep = ids_shuffle[:, :len_keep]

    # Gather visible patches
    x_visible = jnp.take_along_axis(x, ids_keep[..., None], axis=1)

    # Build boolean mask: True = masked (not kept)
    batch_idx = jnp.arange(batch)[:, None]
    mask = jnp.ones((batch, num_patches), dtype=bool)
    mask = mask.at[batch_idx, ids_keep].set(False)

    return x_visible, ids_restore, mask


def unshuffle(x_visible, ids_restore, mask_token, num_patches):
    """Unshuffle visible encoder outputs and fill masked positions with a learned token.

    This is the inverse of ``random_masking``: visible patches (which are in
    shuffled order) are placed back at their original positions, and all other
    positions receive the shared ``mask_token`` embedding.

    Args:
        x_visible:   (batch, len_keep, dim) — encoder outputs for visible patches
        ids_restore: (batch, num_patches)    — inverse permutation from ``random_masking``
        mask_token:  (1, 1, dim) or (dim,)   — learnable mask token
        num_patches: int                      — total patch count

    Returns:
        x_full: (batch, num_patches, dim) — full sequence (visible + mask tokens)
    """
    batch, len_keep, dim = x_visible.shape

    # Normalise mask_token shape to (1, 1, dim)
    if mask_token.ndim == 1:
        mask_token = mask_token[None, None, :]

    # Start with all mask tokens
    x_full = jnp.tile(mask_token, (batch, num_patches, 1))

    # Place visible encoder outputs at shuffled positions 0 .. len_keep-1
    x_full = x_full.at[:, :len_keep, :].set(x_visible)

    # Unshuffle via inverse permutation
    gather_idx = ids_restore[..., None]                     # (batch, N, 1)
    x_full = jnp.take_along_axis(x_full, gather_idx, axis=1)

    return x_full
