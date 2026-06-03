import sys, os
from pathlib import Path as _Path
# 自动将 src 目录加入模块搜索路径，支持从项目根目录直接运行
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

# 必须在 import jax 之前设置：禁用 CUDA 命令缓冲区，避免 CUDA 13.1 驱动与 CUDA 12.x 运行时的图捕获不兼容
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from functools import partial


# 1. 位置编码与 CLS Token（可学习） =============================
def PositionalEncoding(embed_dim):
    """可学习的位置编码（ViT 标准做法）"""
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


# 2. Vision Transformer =========================================
def VisionTransformer(patch_size=4, embed_dim=384, num_layers=7,
                     num_heads=6, mlp_dim=768, dropout_rate=0.2, num_classes=10):
    """Vision Transformer (CIFAR-10)"""
    # 计算分块数量 (CIFAR-10 图像为 32x32)
    num_patches = (32 // patch_size) ** 2
    
    # 分块嵌入层 - 标准实现
    patch_embed = stax.serial(
        Conv(embed_dim, (patch_size, patch_size), 
             strides=(patch_size, patch_size), padding='VALID'),
        Lambda(lambda x: x.reshape(x.shape[0], num_patches, embed_dim))
    )
    
    # 可学习的分类标记
    cls_token = CLSToken(embed_dim)
    
    # 可学习的位置编码
    positional_encoding = PositionalEncoding(embed_dim)
    
    # Transformer编码器
    head_dim = embed_dim // num_heads
    seq_len = num_patches + 1
    # block_size 必须整除 seq_len (flash attention 分块要求)
    # CIFAR patch=4: num_patches=64, seq_len=65, block_size=13
    block_size = seq_len
    encoder = stax.serial(*[
        TransformerEncoderBlock(num_heads, head_dim, embed_dim, mlp_dim, dropout_rate, block_size)
        for _ in range(num_layers)
    ])
    
    # 分类头
    head = stax.serial(
        LayerNorm(),
        Lambda(lambda x: x[:, 0]),  # 提取[CLS]token
        Dense(num_classes)
    )
    
    # 完整模型
    return stax.serial(
        patch_embed,
        cls_token,
        positional_encoding,
        encoder,
        head
    )

# 5. 训练工具函数 ===============================================
def create_train_state(rng, peak_lr=3e-4, total_steps=10000, warmup_steps=1000,
                       weight_decay=0.1, grad_clip_norm=1.0):
    """初始化模型参数和优化器状态（warmup + cosine decay + grad clip）"""
    # 初始化模型
    model_init, model_apply = VisionTransformer(num_classes=10)
    
    # CIFAR-10 输入形状 (batch, H, W, C)
    input_shape = (1, 32, 32, 3)
    _, params, bn_states = model_init(rng, input_shape)
    
    # 学习率调度：linear warmup → cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=peak_lr * 0.01,
    )
    
    # 优化器链：gradient clip → AdamW (with schedule + weight decay)
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = tx.init(params)
    
    return params, bn_states, model_apply, tx, opt_state

# 6. 训练步骤 ===================================================
def make_train_step(apply_fn, tx):
    """创建 JIT 编译的训练步骤"""
    @jax.jit
    def train_step(params, bn_states, opt_state, batch, rng):
        images, labels = batch
        
        # 分裂随机键
        rng, dropout_rng = jax.random.split(rng)
        
        # 标签平滑交叉熵损失（按 batch 取均值）
        def smooth_cross_entropy(logits, labels, smoothing=0.1):
            num_classes = logits.shape[-1]
            smoothed_labels = (1.0 - smoothing) * labels + smoothing / num_classes
            loss_per_sample = -jnp.sum(smoothed_labels * jax.nn.log_softmax(logits), axis=-1)
            return jnp.mean(loss_per_sample)

        # 定义损失函数
        def loss_fn(params):
            logits, _ = apply_fn(params, bn_states, images, rng=dropout_rng, is_training=True)
            loss = smooth_cross_entropy(logits, labels, smoothing=0.1)
            return loss, logits
        
        # 计算梯度和更新
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # 计算准确率
        accuracy = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1)).mean()
        return new_params, bn_states, new_opt_state, loss, accuracy, rng
    
    return train_step

def make_eval_step(apply_fn):
    """创建 JIT 编译的评估步骤"""
    @jax.jit
    def eval_step(params, bn_states, batch):
        images, labels = batch
        logits, _ = apply_fn(params, bn_states, images, is_training=False)  # 评估时不传递rng（dropout 自动关闭）
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        accuracy = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1)).mean()
        return loss, accuracy
    
    return eval_step

# 7. 数据加载 ===================================================
def load_cifar10(batch_size=128):
    """加载 CIFAR-10 数据集，返回 PyTorch DataLoader"""
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
    ])
    
    # 测试数据（仅归一化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
    ])

    # 加载数据集
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    # 创建数据加载器（只创建一次，每 epoch 复用）
    # num_workers=0 必须：JAX 的 CUDA 线程模型不兼容 PyTorch 的 fork worker
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def batch_to_jax(images, labels, num_classes=10):
    """将 PyTorch batch 转为 JAX 数组"""
    images_jax = jnp.asarray(images.numpy())
    labels_onehot = jax.nn.one_hot(labels.numpy(), num_classes=num_classes)
    return images_jax, labels_onehot



def mixup_batch(images, labels, rng, alpha=0.2):
    """MixUp augmentation — 仅对非 one-hot 标签做 mixup"""
    batch_size = images.shape[0]
    rng1, rng2 = jax.random.split(rng)
    lam = jax.random.beta(rng1, alpha, alpha)
    lam = jnp.maximum(lam, 1.0 - lam)  # 保证 lam >= 0.5
    idx = jax.random.permutation(rng2, batch_size)
    mixed_images = lam * images + (1.0 - lam) * images[idx]
    mixed_labels = lam * labels + (1.0 - lam) * labels[idx]
    return mixed_images, mixed_labels

# 8. 主训练循环 ==================================================
if __name__ == "__main__":
    # 初始化
    rng = jax.random.PRNGKey(42)
    batch_size = 128
    learning_rate = 3e-4
    
    # 训练配置
    epochs = 500
    steps_per_epoch = 50000 // batch_size
    test_steps = 10000 // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_epochs = 10
    warmup_steps = warmup_epochs * steps_per_epoch
    
    print("开始创建模型...")
    start_time = time.time()
    
    # 创建训练状态
    params, bn_states, model_apply, tx, opt_state = create_train_state(
        rng, peak_lr=learning_rate, total_steps=total_steps, warmup_steps=warmup_steps
    )
    
    # 创建训练和评估步骤
    train_step = make_train_step(model_apply, tx)
    eval_step = make_eval_step(model_apply)
    
    # 加载数据（只加载一次，复用 DataLoader）
    print("加载 CIFAR-10 数据集...")
    train_loader, test_loader = load_cifar10(batch_size)
    
    model_creation_time = time.time() - start_time
    print(f"模型创建完成，耗时: {model_creation_time:.2f}秒")
    print("开始训练 Vision Transformer (CIFAR-10)...")
    print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, dropout=0.2")
    print(f"LR schedule: warmup={warmup_epochs} epochs, cosine decay to {learning_rate*0.01:.2e}, grad_clip=1.0, weight_decay=0.1")
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        # 准备随机键
        rng, epoch_rng = jax.random.split(rng)
        
        # 训练阶段
        train_losses, train_accs = [], []
        for i, (images, labels) in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            
            # 转为 JAX 格式
            images_jax, labels_jax = batch_to_jax(images, labels)

            # MixUp 正则化 (alpha=0.2，温和混合)
            epoch_rng, mixup_rng = jax.random.split(epoch_rng)
            images_jax, labels_jax = mixup_batch(images_jax, labels_jax, mixup_rng, alpha=0.8)

            # 执行训练步骤
            params, bn_states, opt_state, loss, acc, epoch_rng = train_step(
                params, bn_states, opt_state, (images_jax, labels_jax), epoch_rng
            )
            
            train_losses.append(loss)
            train_accs.append(acc)
            
            # 每100步打印一次
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i}: loss={loss:.4f}, acc={acc*100:.2f}%")
        
        # 评估阶段
        val_losses, val_accs = [], []
        for i, (images, labels) in enumerate(test_loader):
            if i >= test_steps:
                break
            
            # 转为 JAX 格式
            images_jax, labels_jax = batch_to_jax(images, labels)
            loss, acc = eval_step(params, bn_states, (images_jax, labels_jax))
            val_losses.append(loss)
            val_accs.append(acc)
        
        # 计算平均指标
        train_loss = jnp.mean(jnp.array(train_losses))
        train_acc = jnp.mean(jnp.array(train_accs))
        val_loss = jnp.mean(jnp.array(val_losses))
        val_acc = jnp.mean(jnp.array(val_accs))
        
        epoch_time = time.time() - epoch_start
        
        # 保存历史记录
        train_loss_history.append(float(train_loss))
        train_acc_history.append(float(train_acc))
        val_loss_history.append(float(val_loss))
        val_acc_history.append(float(val_acc))
        
        # 打印进度
        print(f"\nEpoch {epoch+1}/{epochs} 完成，耗时: {epoch_time:.2f}秒")
        print(f"  训练损失: {train_loss:.4f}, 准确率: {train_acc*100:.2f}%")
        print(f"  验证损失: {val_loss:.4f}, 准确率: {val_acc*100:.2f}%\n")
    
    print("训练完成!")
    print(f"最终测试准确率: {val_acc_history[-1]*100:.2f}%")
    
    # 保存训练结果
    np.savez('vit_training_history.npz',
             train_loss_history=np.array(train_loss_history),
             train_acc_history=np.array(train_acc_history),
             val_loss_history=np.array(val_loss_history),
             val_acc_history=np.array(val_acc_history))
    
    # 绘制准确率曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), [x * 100 for x in train_acc_history], 'o-', label='训练准确率')
    plt.plot(range(1, epochs+1), [x * 100 for x in val_acc_history], 's-', label='验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('准确率 (%)')
    plt.title('Vision Transformer (CIFAR-10) 训练曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('vit_training_curve.png')
    print("训练曲线已保存为 vit_training_curve.png")
