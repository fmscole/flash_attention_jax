"""
Stage 1: MAE (Masked Autoencoder) Pretraining on CIFAR-10

纯粹的图像重建预训练：
- Encoder: PatchEmbed → +PosEnc → RandomMask → Transformer blocks (no CLS token)
- Decoder: Proj → Unshuffle+MaskToken → +DecoderPosEnc → Decoder blocks → PixelPred
- 损失: MSE only on masked patches

Reference: He et al. "Masked Autoencoders Are Scalable Vision Learners" (2022)

用法:
    python vit_cifar10_mae_pretrain_stax.py

输出:
    ckpt/vit_cifar10_mae_pretrain.pkl       — 预训练参数（Encoder 部分）
    vit_cifar10_mae_pretrain_history.npz     — 训练曲线
    vit_cifar10_mae_pretrain_curve.png        — 可视化
"""

import sys, os
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

if '--xla_gpu_enable_command_buffer' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_enable_command_buffer='

import jax
import jax.numpy as jnp
import optax
from lib import stax_plus as stax
from lib.stax_plus import (
    Dense, Relu, Dropout, FanOut, FanInConcat, FanInSum, Identity,
    Lambda, LayerNorm, Flatten, Conv
)
from transformers.transformer_flash_v1 import TransformerEncoderBlock
from transformers.mae_utils import random_masking, unshuffle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time, pickle
import numpy as np
from functools import partial


# ═══════════════════════════════════════════════════════════════════
# 基础组件（与 supervised ViT 共用）
# ═══════════════════════════════════════════════════════════════════

def PositionalEncoding(embed_dim):
    """可学习的位置编码"""
    def init_fun(rng, input_shape):
        _, seq_len, _ = input_shape
        pe = jax.random.normal(rng, (1, seq_len, embed_dim)) * 0.02
        return input_shape, pe
    def apply_fun(params, inputs, **kwargs):
        return inputs + params[:, :inputs.shape[1], :]
    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# MAE 预训练模型（Encoder—Decoder）
# ═══════════════════════════════════════════════════════════════════

def MAEPretrainingModel(
    patch_size=4,
    embed_dim=384,
    num_layers=7,
    num_heads=6,
    mlp_dim=768,
    dropout_rate=0.2,
    mask_ratio=0.6,
    decoder_depth=2,
    decoder_embed=192,
    decoder_num_heads=3,
):
    """MAE 预训练模型：patch 重建，无 CLS token。

    Returns:
        (init_fun, apply_fun) — Stax 风格的层对
    """
    num_patches = (32 // patch_size) ** 2                       # 64
    pixel_dim = patch_size * patch_size * 3                     # 48
    head_dim = embed_dim // num_heads                           # 64
    dec_head_dim = decoder_embed // decoder_num_heads           # 64

    # ── Encoder 子模块 ────────────────────────────────────────
    patch_embed = stax.serial(
        Conv(embed_dim, (patch_size, patch_size),
             strides=(patch_size, patch_size), padding='VALID'),
        Lambda(lambda x: x.reshape(x.shape[0], num_patches, embed_dim))
    )
    encoder_pe = PositionalEncoding(embed_dim)

    enc_block_size = num_patches  # 64, forward 时按实际 seq_len clamp
    encoder_blocks = [
        TransformerEncoderBlock(num_heads, head_dim, embed_dim, mlp_dim,
                                dropout_rate, enc_block_size)
        for _ in range(num_layers)
    ]
    encoder = stax.serial(*encoder_blocks)

    # ── Decoder 子模块 ────────────────────────────────────────
    decoder_proj = Dense(decoder_embed)
    mask_token_init = lambda rng, _shape: (
        jax.random.normal(rng, (decoder_embed,)) * 0.02
    )
    decoder_pe = PositionalEncoding(decoder_embed)

    dec_block_size = num_patches
    decoder_blocks = [
        TransformerEncoderBlock(decoder_num_heads, dec_head_dim,
                                decoder_embed, decoder_embed * 4,
                                dropout_rate, dec_block_size)
        for _ in range(decoder_depth)
    ]
    decoder = stax.serial(*decoder_blocks)
    decoder_norm = LayerNorm()
    decoder_pred = Dense(pixel_dim)

    # ── init_fun ──────────────────────────────────────────────
    def init_fun(rng, input_shape):
        (rng_patch, rng_enc_pe, rng_enc, rng_proj,
         rng_mask, rng_dec_pe, rng_dec, rng_dec_norm, rng_pred) = \
            jax.random.split(rng, 9)

        # Patch embedding
        _, patch_params, patch_bn = patch_embed[0](rng_patch, input_shape)
        patch_out_shape = (input_shape[0], num_patches, embed_dim)

        # Encoder PE
        _, enc_pe_params = encoder_pe[0](rng_enc_pe, patch_out_shape)

        # Encoder (init with approx len_keep for shape reference)
        approx_keep = int(num_patches * (1.0 - mask_ratio))
        enc_in_shape = (input_shape[0], approx_keep, embed_dim)
        _, enc_params, enc_bn = encoder[0](rng_enc, enc_in_shape)

        # Decoder proj
        _, proj_params = decoder_proj[0](rng_proj, (input_shape[0], approx_keep, embed_dim))

        # Mask token
        mask_token = mask_token_init(rng_mask, (decoder_embed,))

        # Decoder PE
        dec_in_shape = (input_shape[0], num_patches, decoder_embed)
        _, dec_pe_params = decoder_pe[0](rng_dec_pe, dec_in_shape)

        # Decoder blocks
        _, dec_params, dec_bn = decoder[0](rng_dec, dec_in_shape)

        # Decoder LN
        _, dec_norm_params = decoder_norm[0](rng_dec_norm, dec_in_shape)

        # Prediction head
        _, pred_params = decoder_pred[0](rng_pred, dec_in_shape)

        params = {
            'patch_embed': patch_params,
            'encoder_pe': enc_pe_params,
            'encoder': enc_params,
            'decoder_proj': proj_params,
            'mask_token': mask_token,
            'decoder_pe': dec_pe_params,
            'decoder': dec_params,
            'prediction_head': pred_params,
            'decoder_norm': dec_norm_params,
        }

        bn_states = {
            'patch_embed': patch_bn,
            'encoder': enc_bn,
            'decoder': dec_bn,
        }

        return input_shape, params, bn_states

    # ── apply_fun ─────────────────────────────────────────────
    def apply_fun(params, bn_states, inputs,
                  rng=None, mask_rng=None, is_training=True, **kwargs):
        p = params
        b = bn_states

        # 1. Patch embedding
        x, new_patch_bn = patch_embed[1](p['patch_embed'], b['patch_embed'],
                                          inputs, rng=rng, is_training=is_training)
        # 2. Encoder positional encoding (before masking — crucial!)
        x = encoder_pe[1](p['encoder_pe'], x)
        # 3. Random masking
        if is_training and mask_rng is not None:
            x_vis, ids_restore, mask = random_masking(x, mask_ratio, mask_rng)
        else:
            x_vis, ids_restore, mask = x, None, None

        # 4. Encoder (no CLS token)
        enc_out, new_enc_bn = encoder[1](p['encoder'], b['encoder'],
                                          x_vis, rng=rng, is_training=is_training)
        # enc_out: (batch, len_keep, embed_dim)

        # 5. Decoder
        if is_training and mask_rng is not None:
            dec_in = decoder_proj[1](p['decoder_proj'], enc_out)
            dec_in = unshuffle(dec_in, ids_restore, p['mask_token'], num_patches)
            dec_in = decoder_pe[1](p['decoder_pe'], dec_in)
            dec_out, new_dec_bn = decoder[1](p['decoder'], b['decoder'],
                                              dec_in, rng=rng, is_training=is_training)
            dec_out = decoder_norm[1](p['decoder_norm'], dec_out)
            recon = decoder_pred[1](p['prediction_head'], dec_out)
        else:
            recon = None
            mask = None
            new_dec_bn = b['decoder']

        new_bn = {
            'patch_embed': new_patch_bn,
            'encoder': new_enc_bn,
            'decoder': new_dec_bn,
        }

        return (recon, mask), new_bn

    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# 辅助：提取 patches 作为重建目标
# ═══════════════════════════════════════════════════════════════════

def extract_patches(images, patch_size=4):
    """将 (batch, H, W, C) 图像拆分为展平的 patch 向量。"""
    batch, h, w, c = images.shape
    nh, nw = h // patch_size, w // patch_size
    patches = images.reshape(batch, nh, patch_size, nw, patch_size, c)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(batch, nh * nw, patch_size * patch_size * c)
    return patches


# ═══════════════════════════════════════════════════════════════════
# 训练状态
# ═══════════════════════════════════════════════════════════════════

def create_pretrain_state(rng, peak_lr=1.5e-4, total_steps=10000, warmup_steps=1000,
                          weight_decay=0.05, grad_clip_norm=1.0):
    """初始化 MAE 预训练参数和优化器。"""
    model_init, model_apply = MAEPretrainingModel()

    input_shape = (1, 32, 32, 3)
    _, params, bn_states = model_init(rng, input_shape)

    # 参数量统计
    enc_param_count = sum(
        p.size for p in jax.tree_util.tree_leaves(
            {k: v for k, v in params.items() if k not in ('decoder_proj', 'mask_token',
                                                           'decoder_pe', 'decoder',
                                                           'prediction_head', 'decoder_norm')}
        )
    )
    dec_param_count = sum(
        p.size for p in jax.tree_util.tree_leaves(
            {k: v for k, v in params.items() if k in ('decoder_proj', 'mask_token',
                                                       'decoder_pe', 'decoder',
                                                       'prediction_head', 'decoder_norm')}
        )
    )
    total_params = enc_param_count + dec_param_count
    print(f"MAE 预训练模型参数: Encoder={enc_param_count:,}  Decoder={dec_param_count:,}  总计={total_params:,}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=peak_lr * 0.01,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = tx.init(params)

    return params, bn_states, model_apply, tx, opt_state


# ═══════════════════════════════════════════════════════════════════
# 训练步骤
# ═══════════════════════════════════════════════════════════════════

def make_pretrain_step(apply_fn, tx):
    """创建 JIT 编译的 MAE 预训练步骤。"""

    @jax.jit
    def train_step(params, bn_states, opt_state, images, rng):
        rng, dropout_rng, mask_rng = jax.random.split(rng, 3)

        def loss_fn(params):
            (recon, mask), new_bn = apply_fn(
                params, bn_states, images,
                rng=dropout_rng, mask_rng=mask_rng, is_training=True
            )
            target = extract_patches(images, patch_size=4)
            diff = (recon - target) ** 2                              # (batch, 64, 48)
            mask_float = mask.astype(jnp.float32)                     # (batch, 64)
            # MSE only on masked positions
            loss = (diff.mean(axis=-1) * mask_float).sum() / (mask_float.sum() + 1e-8)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, bn_states, new_opt_state, loss, rng

    return train_step


# ═══════════════════════════════════════════════════════════════════
# 数据加载（轻量增强，无 MixUp）
# ═══════════════════════════════════════════════════════════════════

def load_cifar10_pretrain(batch_size=128):
    """加载 CIFAR-10 用于 MAE 预训练（无 MixUp，轻量增强）。

    MAE 预训练不需要强增强：masking 本身提供了足够的正则化。
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))  # (C,H,W) → (H,W,C)
    ])

    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader


def batch_to_jax(images, labels=None):
    """将 PyTorch batch 转为 JAX 数组（pretrain 不需要 labels）。"""
    return jnp.asarray(images.numpy())


# ═══════════════════════════════════════════════════════════════════
# 主训练循环
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1: MAE Pretraining (CIFAR-10)")
    print("Encoder: {}-layer ViT, no CLS | Decoder: {}-layer light".format(7, 2))
    print("=" * 60)

    rng = jax.random.PRNGKey(42)
    batch_size = 128

    # ── 训练配置 ──
    epochs = 100
    steps_per_epoch = 50000 // batch_size                     # ~390
    total_steps = epochs * steps_per_epoch
    warmup_epochs = 20
    warmup_steps = warmup_epochs * steps_per_epoch

    print(f"\n超参数: epochs={epochs}, batch_size={batch_size}, steps_per_epoch={steps_per_epoch}")
    print(f"LR: peak=1.5e-4, warmup={warmup_epochs} epochs, cosine→1.5e-6")
    print(f"Optimizer: AdamW(weight_decay=0.05), grad_clip=1.0")
    print(f"MAE: mask_ratio=0.6, decoder=2×192d\n")

    print("创建模型...")
    start_time = time.time()

    params, bn_states, model_apply, tx, opt_state = create_pretrain_state(
        rng, peak_lr=1.5e-4, total_steps=total_steps, warmup_steps=warmup_steps
    )

    train_step = make_pretrain_step(model_apply, tx)

    print("加载 CIFAR-10...")
    train_loader = load_cifar10_pretrain(batch_size)

    model_time = time.time() - start_time
    print(f"就绪，耗时: {model_time:.2f}s\n")

    loss_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        rng, epoch_rng = jax.random.split(rng)

        losses = []
        for i, (images, _labels) in enumerate(train_loader):
            if i >= steps_per_epoch:
                break

            images_jax = batch_to_jax(images)
            params, bn_states, opt_state, loss, epoch_rng = train_step(
                params, bn_states, opt_state, images_jax, epoch_rng
            )
            losses.append(float(loss))

            if i % 100 == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs} | Step {i:4d} | loss={loss:.6f}")

        avg_loss = np.mean(losses)
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{epochs} 完成 | loss={avg_loss:.6f} | 耗时={epoch_time:.1f}s\n")

    print("预训练完成！")

    # ── 保存 Encoder 权重（供 Stage 2 fine-tune 使用） ──
    os.makedirs('ckpt', exist_ok=True)
    encoder_weights = {
        'patch_embed': params['patch_embed'],
        'encoder_pe': params['encoder_pe'],
        'encoder': params['encoder'],
    }
    with open('ckpt/vit_cifar10_mae_pretrain.pkl', 'wb') as f:
        pickle.dump(encoder_weights, f)
    print("Encoder 权重已保存到 ckpt/vit_cifar10_mae_pretrain.pkl")

    # ── 保存损失曲线 ──
    np.savez('vit_cifar10_mae_pretrain_history.npz',
             loss=np.array(loss_history))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), loss_history, 'b-', linewidth=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('MAE Pretraining — CIFAR-10')
    plt.grid(True, alpha=0.3)
    plt.savefig('vit_cifar10_mae_pretrain_curve.png', dpi=150)
    print("训练曲线已保存到 vit_cifar10_mae_pretrain_curve.png")
