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

## License

MIT
