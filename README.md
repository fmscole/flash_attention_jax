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

基于 Flash Attention 的 Encoder-Decoder Transformer，使用 AiChallenger 机器翻译数据集（1000 万句对）进行训练。

### 数据集

**AiChallenger 机器翻译数据集** — 通用领域中英平行语料，约 1000 万句对。

- 下载地址: [AiChallenger 机器翻译](https://aistudio.baidu.com/datasetdetail/220848)
- 本地路径: `H:\data_set\AiChallenger\`
- 格式: 已对齐的 `train.en` / `train.zh`（各 1000 万行，行号对应）
- 验证集: `valid.en-zh.en.sgm` / `valid.en-zh.zh.sgm`（各 8000 句）
- 清洗脚本: `src/translation/prepare_ai_challenger.py`

```bash
# 从原始文件生成训练集 + 验证集 TSV（默认输出 300 万句对训练 + 8000 句对验证）
python src/translation/prepare_ai_challenger.py
```

输出:
| 文件 | 内容 | 大小 |
|------|------|------|
| `data/ai_challenger_zh_en.tsv` | 训练集 (en\tzh) | ~232 MB |
| `data/ai_challenger_valid_zh_en.tsv` | 官方验证集 (en\tzh) | ~887 KB |

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
# 训练（需先运行 prepare_ai_challenger.py 生成 TSV）
python src/translation/translate_stax_flash.py
```

### 训练细节

| 配置 | 值 |
|------|-----|
| 数据集 | AiChallenger 机器翻译（通用领域，~1000 万句对） |
| 训练模式 | epoch-based |
| Max epochs | 10000（早停触发终止） |
| Optimizer | AdamW (weight_decay=1e-4) |
| Peak LR | 3e-5, warmup 4000 steps, cosine decay → 1e-5 |
| Batch size | 64 |
| Max length | 64 |
| Grad clip | global norm 1.0 |
| 保存 | 每 epoch 自动保存 — `model_ai_challenger_zh_en.pkl`（含优化器状态），重启自动恢复 |
| 训练曲线 | 每 epoch 更新绘制 → `translate_jax/model_ai_challenger_zh_en_curve.png` |
| 验证 | 每 10 epoch — 官方验证集 8000 句（非随机切分），通过时保存 `*_best.pkl` |
| 早停 | val loss 连续 3 次验证不降即停（约 30 epoch） |

推理时使用贪心解码，每个时间步取 argmax，遇到 `<eos>` 停止。

### 训练进展

**Epoch 1**（约 2 小时，batch_size=64，46875 batches）:

| 指标 | 值 |
|------|-----|
| Loss（batch 0） | 10.90 |
| Loss（batch 500） | 9.95 |
| Loss（batch 44500–46500） | 2.90 ~ 3.24 |
| Grad norm | 3.0 ~ 4.0 |

第 1 个 epoch 结束时的翻译示例（贪心解码）:

```
EN: they planted roses along the fence every spring morning
ZH: 每天早上他们都在<unk><unk>了玫瑰

EN: he missed the train because his alarm never rang
ZH: 他错过了火车因为他的警报没有

EN: he is also very famous in japan
ZH: 他也是日本人

EN: i don 't expect anything from you
ZH: 我不希望你能得到什么

EN: i found it easy to speak english
ZH: 我知道说英语很容易

EN: she opened the old wooden box and found a letter inside
ZH: 她打开了旧的<unk>在里面找到了一封信

EN: time passes quickly when we 're doing something we like .
ZH: 我们现在的时候需要的时候时间就快

EN: tom needs to study more if he hopes to pass this class .
ZH: 如果他希望通过这个课需要更多学习
```

## License

MIT
