"""
Stage 2: MAE Fine-tuning on CIFAR-10

加载 Stage 1 预训练的 Encoder 权重，添加 CLS token + 分类头，用全图进行监督微调。

用法:
    python vit_cifar10_mae_finetune_stax.py

依赖:
    ckpt/vit_cifar10_mae_pretrain.pkl  — Stage 1 输出

Reference: He et al. "Masked Autoencoders Are Scalable Vision Learners" (2022)
"""

import sys, os, pickle
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
from transformers.mae_utils import random_masking, unshuffle  # unused here, import for consistency
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time, pickle
import numpy as np
from functools import partial


# ═══════════════════════════════════════════════════════════════════
# 基础组件
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


def CLSToken(embed_dim):
    """可学习的分类 token"""
    def init_fun(rng, input_shape):
        cls = jax.random.normal(rng, (1, 1, embed_dim)) * 0.02
        out_shape = (input_shape[0], input_shape[1] + 1, embed_dim)
        return out_shape, cls
    def apply_fun(params, inputs, **kwargs):
        batch = inputs.shape[0]
        cls_tokens = jnp.tile(params, (batch, 1, 1))
        return jnp.concatenate([cls_tokens, inputs], axis=1)
    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# Fine-tune 模型：Encoder（预训练权重）+ CLS + 分类头
# ═══════════════════════════════════════════════════════════════════

def ViTFinetuneModel(
    patch_size=4,
    embed_dim=384,
    num_layers=7,
    num_heads=6,
    mlp_dim=768,
    dropout_rate=0.0,           # 微调阶段建议低 dropout
    num_classes=10,
    encoder_stochastic_depth=0.0,
):
    """Vision Transformer for fine-tuning (encodes full image, no decoder).

    Returns:
        (init_fun, apply_fun) — Stax 风格的层对
        init_fun 接受 pretrained_params 字典来初始化 Encoder
    """
    num_patches = (32 // patch_size) ** 2
    head_dim = embed_dim // num_heads

    # ── 子模块 ────────────────────────────────────────────────
    patch_embed = stax.serial(
        Conv(embed_dim, (patch_size, patch_size),
             strides=(patch_size, patch_size), padding='VALID'),
        Lambda(lambda x: x.reshape(x.shape[0], num_patches, embed_dim))
    )

    encoder_pe = PositionalEncoding(embed_dim)
    cls_token = CLSToken(embed_dim)

    block_size = num_patches + 1  # 65
    encoder_blocks = [
        TransformerEncoderBlock(num_heads, head_dim, embed_dim, mlp_dim,
                                dropout_rate, block_size)
        for _ in range(num_layers)
    ]
    encoder = stax.serial(*encoder_blocks)

    classification_head = stax.serial(
        LayerNorm(),
        Lambda(lambda x: x[:, 0]),     # 提取 [CLS] token
        Dense(num_classes)
    )

    # ── init_fun ──────────────────────────────────────────────
    # 接受 pretrained_encoder 参数来初始化
    def init_fun(rng, input_shape, pretrained_encoder=None):
        rng_patch, rng_pe, rng_cls, rng_enc, rng_head = jax.random.split(rng, 5)

        _, patch_params, patch_bn = patch_embed[0](rng_patch, input_shape)
        patch_out_shape = (input_shape[0], num_patches, embed_dim)

        _, pe_params = encoder_pe[0](rng_pe, patch_out_shape)
        _, cls_params = cls_token[0](rng_cls, patch_out_shape)

        enc_input_shape = (input_shape[0], num_patches + 1, embed_dim)
        _, enc_params, enc_bn = encoder[0](rng_enc, enc_input_shape)

        _, head_params, head_bn = classification_head[0](rng_head, enc_input_shape)

        # 如果提供了预训练权重，覆盖 Encoder 部分
        if pretrained_encoder is not None:
            print("加载预训练 Encoder 权重...")
            pt = pretrained_encoder
            patch_params = pt['patch_embed']
            pe_params = pt['encoder_pe']
            enc_params = pt['encoder']

        params = {
            'patch_embed': patch_params,
            'encoder_pe': pe_params,
            'cls_token': cls_params,
            'encoder': enc_params,
            'classification_head': head_params,
        }

        bn_states = {
            'patch_embed': patch_bn,
            'encoder': enc_bn,
            'classification_head': head_bn,
        }

        return input_shape, params, bn_states

    # ── apply_fun ─────────────────────────────────────────────
    def apply_fun(params, bn_states, inputs, rng=None, is_training=True, **kwargs):
        p = params
        b = bn_states

        # 1. Patch embedding
        x, new_patch_bn = patch_embed[1](p['patch_embed'], b['patch_embed'],
                                          inputs, rng=rng, is_training=is_training)
        # 2. Positional encoding
        x = encoder_pe[1](p['encoder_pe'], x)
        # 3. Prepend CLS token
        x = cls_token[1](p['cls_token'], x)
        # 4. Encoder (full sequence: 65 tokens)
        enc_out, new_enc_bn = encoder[1](p['encoder'], b['encoder'],
                                          x, rng=rng, is_training=is_training)
        # 5. Classification head
        logits, new_head_bn = classification_head[1](
            p['classification_head'], b['classification_head'],
            enc_out, rng=rng, is_training=is_training
        )

        new_bn = {
            'patch_embed': new_patch_bn,
            'encoder': new_enc_bn,
            'classification_head': new_head_bn,
        }

        return logits, new_bn

    return init_fun, apply_fun


# ═══════════════════════════════════════════════════════════════════
# 训练状态
# ═══════════════════════════════════════════════════════════════════

def create_finetune_state(rng, pretrained_path,
                          peak_lr=1e-4, total_steps=10000, warmup_steps=1000,
                          weight_decay=0.05, grad_clip_norm=1.0):
    """加载预训练 Encoder 权重，创建 fine-tune 模型。

    使用统一 AdamW 优化器，全部参数用 peak_lr。
    """
    # ── 加载预训练权重 ──
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"预训练权重未找到: {pretrained_path}\n"
            f"请先运行 Stage 1: python vit_cifar10_mae_pretrain_stax.py"
        )

    with open(pretrained_path, 'rb') as f:
        pretrained_encoder = pickle.load(f)

    enc_count = sum(p.size for p in jax.tree_util.tree_leaves(pretrained_encoder))
    print(f"已加载预训练 Encoder 权重 ({enc_count:,} 参数)")

    # ── 初始化 fine-tune 模型 ──
    model_init, model_apply = ViTFinetuneModel(dropout_rate=0.1)
    input_shape = (1, 32, 32, 3)
    _, params, bn_states = model_init(rng, input_shape, pretrained_encoder=pretrained_encoder)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Fine-tune 模型参数总量: {total_params:,}")

    # ── 优化器（统一学习率，与 MAE 论文 fine-tune 一致）──
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

def make_finetune_step(apply_fn, tx):
    """创建 JIT 编译的 fine-tune 训练步骤。"""

    @jax.jit
    def train_step(params, bn_states, opt_state, batch, rng):
        images, labels = batch
        rng, dropout_rng = jax.random.split(rng)

        def smooth_cross_entropy(logits, labels, smoothing=0.1):
            num_classes = logits.shape[-1]
            smoothed = (1.0 - smoothing) * labels + smoothing / num_classes
            return -jnp.sum(smoothed * jax.nn.log_softmax(logits), axis=-1).mean()

        def loss_fn(params):
            logits, _ = apply_fn(
                params, bn_states, images,
                rng=dropout_rng, is_training=True
            )
            loss = smooth_cross_entropy(logits, labels, smoothing=0.1)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        accuracy = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1)).mean()
        return new_params, bn_states, new_opt_state, loss, accuracy, rng

    return train_step


def make_eval_step(apply_fn):
    """创建 JIT 编译的评估步骤（全图，无 masking）。"""

    @jax.jit
    def eval_step(params, bn_states, batch):
        images, labels = batch
        logits, _ = apply_fn(params, bn_states, images, is_training=False)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        accuracy = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1)).mean()
        return loss, accuracy

    return eval_step


# ═══════════════════════════════════════════════════════════════════
# 数据加载（强增强 + MixUp，与 supervised ViT 一致）
# ═══════════════════════════════════════════════════════════════════

def load_cifar10(batch_size=128):
    """加载 CIFAR-10，带 RandAugment + MixUp 准备。"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))
    ])

    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def batch_to_jax(images, labels, num_classes=10):
    """将 PyTorch batch 转为 JAX 数组（one-hot 标签）。"""
    return jnp.asarray(images.numpy()), jax.nn.one_hot(labels.numpy(), num_classes)


def mixup_batch(images, labels, rng, alpha=0.2):
    """MixUp 数据增强。"""
    batch_size = images.shape[0]
    rng1, rng2 = jax.random.split(rng)
    lam = jax.random.beta(rng1, alpha, alpha)
    lam = jnp.maximum(lam, 1.0 - lam)
    idx = jax.random.permutation(rng2, batch_size)
    mixed_images = lam * images + (1.0 - lam) * images[idx]
    mixed_labels = lam * labels + (1.0 - lam) * labels[idx]
    return mixed_images, mixed_labels


# ═══════════════════════════════════════════════════════════════════
# 主训练循环
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 2: MAE Fine-tuning (CIFAR-10)")
    print("=" * 60)

    rng = jax.random.PRNGKey(123)  # 不同于 pretrain 的 seed
    batch_size = 128

    # ── 训练配置 ──
    epochs = 200
    steps_per_epoch = 50000 // batch_size
    test_steps = 10000 // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_epochs = 5
    warmup_steps = warmup_epochs * steps_per_epoch

    # ── 超参数 ──
    peak_lr = 1e-4            # 全部参数统一用较小 LR（Encoder 已充分预训练）
    weight_decay = 0.05

    print(f"fine-tune epochs={epochs}, warmup={warmup_epochs}, batch={batch_size}")
    print(f"LR={peak_lr}, weight_decay={weight_decay}")
    print(f"MixUp alpha=0.8, label_smoothing=0.1\n")

    print("加载预训练权重并创建模型...")
    start_time = time.time()

    pretrained_path = os.path.join(
        os.path.dirname(__file__), '../../ckpt/vit_cifar10_mae_pretrain.pkl'
    )
    pretrained_path = os.path.normpath(pretrained_path)

    params, bn_states, model_apply, tx, opt_state = create_finetune_state(
        rng, pretrained_path,
        peak_lr=peak_lr, total_steps=total_steps, warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )

    train_step = make_finetune_step(model_apply, tx)
    eval_step = make_eval_step(model_apply)

    print("加载 CIFAR-10...")
    train_loader, test_loader = load_cifar10(batch_size)

    model_time = time.time() - start_time
    print(f"就绪，耗时: {model_time:.2f}s\n")

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        rng, epoch_rng = jax.random.split(rng)

        # ── 训练 ──
        train_losses, train_accs = [], []
        for i, (images, labels) in enumerate(train_loader):
            if i >= steps_per_epoch:
                break

            images_jax, labels_jax = batch_to_jax(images, labels)

            # MixUp
            epoch_rng, mixup_rng = jax.random.split(epoch_rng)
            images_jax, labels_jax = mixup_batch(images_jax, labels_jax, mixup_rng, alpha=0.8)

            params, bn_states, opt_state, loss, acc, epoch_rng = train_step(
                params, bn_states, opt_state, (images_jax, labels_jax), epoch_rng
            )

            train_losses.append(float(loss))
            train_accs.append(float(acc))

            if i % 100 == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs} | Step {i:4d} | loss={loss:.4f} acc={acc*100:.2f}%")

        # ── 评估 ──
        val_losses, val_accs = [], []
        for i, (images, labels) in enumerate(test_loader):
            if i >= test_steps:
                break
            images_jax, labels_jax = batch_to_jax(images, labels)
            loss, acc = eval_step(params, bn_states, (images_jax, labels_jax))
            val_losses.append(float(loss))
            val_accs.append(float(acc))

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        epoch_time = time.time() - epoch_start

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs} — "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
              f"{epoch_time:.1f}s\n")

    print("Fine-tune 完成！")
    print(f"最终测试准确率: {val_acc_history[-1]*100:.2f}%")

    # ── 保存模型参数（供实验 A self-distill 使用）──
    os.makedirs('../../ckpt', exist_ok=True)
    model_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '../../ckpt/vit_cifar10_mae_finetune.pkl'
    ))
    with open(model_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"模型参数已保存到 {model_path}")

    # ── 保存训练曲线 ──
    np.savez('vit_cifar10_mae_finetune_history.npz',
             train_loss=np.array(train_loss_history),
             train_acc=np.array(train_acc_history),
             val_loss=np.array(val_loss_history),
             val_acc=np.array(val_acc_history))

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, epochs+1), [x*100 for x in train_acc_history], 'o-', label='Train', markersize=2)
    ax1.plot(range(1, epochs+1), [x*100 for x in val_acc_history], 's-', label='Val', markersize=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('MAE Fine-tune — Classification Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, epochs+1), train_loss_history, label='Train Loss')
    ax2.plot(range(1, epochs+1), val_loss_history, label='Val Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vit_cifar10_mae_finetune_curve.png', dpi=150)
    print("曲线已保存到 vit_cifar10_mae_finetune_curve.png")
