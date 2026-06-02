# Flash Attention JAX

基于 JAX 的 Flash Attention 实现，包含 v1 和 v2 (FlashAttention-2) 两个版本，
以及基于它们的 Transformer、Vision Transformer (ViT) 和 翻译模型 示例应用。

实现参考了 [lucidrains/flash-attention-jax](https://github.com/lucidrains/flash-attention-jax)。

## 核心模块

| 模块 | 说明 |
|------|------|
| `flash_attention/` | **核心** — Flash Attention v1 & v2，自定义 VJP |
| `lib/` | Stax 扩展层（Dense, Dropout, LayerNorm, Conv, BatchNorm 等） |
| `transformers/` | 基于 Flash Attention 的 Transformer 编码器/解码器 |
| `translation/` | 翻译模型示例（EN→ZH，基于 Transformer + Flash Attention） |
| `vit/` | Vision Transformer 示例（含 MAE 预训练）— 单阶段 CIFAR-10 90~91%，两阶段 MAE+微调 **94.92%** |

## Flash Attention

- **v1** (`flash_attention_v1.py`): 标准实现，Q-block outer / KV-block inner 循环
- **v2** (`flash_attention_v2.py`): FlashAttention-2，KV-block outer / Q-block inner 循环
  反向传播循环顺序优化（对应 FA2 论文 §3.2）

两者前向/反向数值一致（差异 < 1e-5）。

### 用法

```python
from flash_attention import flash_attention_v1, flash_attention_v2

# v1
output = flash_attention_v1(q, k, v, block_size_q=128, block_size_kv=128)

# v2
output = flash_attention_v2(q, k, v, causal=True, block_size_q=128, block_size_kv=128)
```

### 数值稳定性

分块模式下 (block_size < seq_len) 使用统一的 `_MIN_NORMALIZER=1e-12` 确保
前向归一化和反向 log-sum-exp 恢复权重完全一致，消除梯度漂移。

## Vision Transformer (CIFAR-10)

三个示例脚本，覆盖从零训练到两阶段自监督微调的完整流程。

### 脚本对比

| 脚本 | 方法 | 准确率 | 前置依赖 |
|------|------|--------|----------|
| `vit_cifar10_stax.py` | 端到端监督训练 | 90~91% | 无 |
| `vit_cifar10_mae_pretrain_stax.py` | Stage 1: MAE 自监督预训练（重建 masked patches） | — | 无 |
| `vit_cifar10_mae_finetune_stax.py` | Stage 2: 加载预训练 Encoder + 分类头微调 | **94.92%** | `ckpt/vit_cifar10_mae_pretrain.pkl` |

### 模型架构

三者共享相同的 backbone 设计（CIFAR-10 图像 32×32）：

- **Patch Embed**: Conv 4×4, stride 4 → 64 patches（每块 4×4×3=48 维）
- **Embed dim**: 384, **Heads**: 6, **Layers**: 7, **MLP dim**: 768
- **位置编码**: 可学习（非正弦固定）
- **注意力**: Flash Attention v1（通过 `TransformerEncoderBlock`）

### 用法

```bash
# 1. 端到端监督训练（约 500 epoch）
python src/vit/vit_cifar10_stax.py

# 2. MAE 自监督预训练（100 epoch，仅重建损失，不需标签）
python src/vit/vit_cifar10_mae_pretrain_stax.py

# 3. 加载预训练权重微调（100 epoch，需先完成 step 2）
python src/vit/vit_cifar10_mae_finetune_stax.py
```

### 训练细节

| 配置 | 监督 (vit_cifar10_stax) | MAE 预训练 | MAE 微调 |
|------|------------------------|------------|----------|
| Epochs | 500 | 100 | 100 |
| Optimizer | AdamW | AdamW | AdamW |
| Peak LR | 3e-4 | 1.5e-4 | 1e-4 |
| LR schedule | warmup 10ep + cosine decay | warmup 20ep + cosine decay | warmup 20ep + cosine decay |
| Weight decay | 0.1 | 0.05 | 0.05 |
| Batch size | 128 | 128 | 128 |
| Dropout | 0.2 | — | 0.1 |
| Data aug | RandAugment + MixUp(α=0.8) | RandomCrop + Flip | RandAugment + MixUp(α=0.8) |
| Label smoothing | 0.1 | — | 0.1 |

**对比**: 单阶段监督训练可达 90–91%；两阶段 MAE 自监督预训练 + 微调可达 **94.92%**（高出约 4–5 个百分点）。

## 翻译模型 (EN→ZH)

基于 Flash Attention 的 Encoder-Decoder Transformer，使用联合国中英平行语料库进行训练。

### 数据集

**UNv1.0.en-zh** — 联合国官方文件平行语料（英中方向），约 2000 万句对。

- 官网: https://conferences.unite.un.org/UNCorpus/Home/DownloadOverview
- 国内镜像: https://aistudio.baidu.com/datasetdetail/163038

下载后解压到 `E:\data\UNv1.0.en-zh\`，包含两个文件：
```
UNv1.0.en-zh.en   — 英文原文（约 2.3 GB，约 2000 万行）
UNv1.0.en-zh.zh   — 中文译文（约 1.8 GB）
```

### 模型架构

| 组件 | 配置 |
|------|------|
| Encoder / Decoder | 各 6 层 Pre-LN Transformer |
| Attention | Flash Attention v1（通过 `Transformer` 封装） |
| Hidden dim | 512 (8 heads × 64 head_dim) |
| FFN dim | 2048 |
| 词表 | 各 55,000（BPE-free，基于词频构建） |
| 特殊标记 | `<pad>`, `<sos>`, `<eos>`, `<unk>` |

### 用法

```bash
# 训练（配置已内置于 translate_stax_flash.py）
python src/translation/translate_stax_flash.py
```

### 训练细节

| 配置 | 值 |
|------|-----|
| 训练模式 | epoch-based（每 epoch 流式读取 ~200K 句随机切片） |
| Max epochs | 200 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Peak LR | 1e-4, warmup 4000 steps, cosine decay → 1e-5 |
| Batch size | 64 |
| Max length | 64 |
| Grad clip | global norm 1.0 |
| 保存 | **每 epoch 自动保存** — `model_un_en_zh.pkl`，重启自动加载 |
| 验证 | **每 10 epoch** — 通过时保存 `*_best.pkl` |
| 训练曲线 | **每 epoch 绘制** → `translate_jax/training_curve.png`（替换旧图） |
| 早停 | val loss 连续 5 次验证不降即停 |
| 词表 | 从前 500K 行采样构建，min_freq=3 |
| 验证集 | 末尾 5000 句（固定） |
| 训练数据 | 流式 `UNIterableDataset`（避免 OOM） |
| 检查点 | `translate_jax/model_un_en_zh.pkl` + `*_best.pkl` |

推理时使用贪心解码，每个时间步取 argmax，遇到 `<eos>` 停止。

## License

MIT
