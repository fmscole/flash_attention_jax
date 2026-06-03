import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import unittest
from flash_attention import flash_attention_v1
class TestFlashAttention(unittest.TestCase):
    def setUp(self):
        # 设置随机种子
        self.key = jax.random.PRNGKey(0)
        
        # 基本参数
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 64
        self.head_dim = 64
        
        # 创建输入数据
        self.q = jax.random.normal(self.key, (self.batch_size * self.num_heads, self.seq_len, self.head_dim))
        self.k = jax.random.normal(self.key, (self.batch_size * self.num_heads, self.seq_len, self.head_dim))
        self.v = jax.random.normal(self.key, (self.batch_size * self.num_heads, self.seq_len, self.head_dim))
        
        # 创建掩码
        self.padding_mask = jnp.ones((self.batch_size, self.seq_len), dtype=bool)
        self.padding_mask = jnp.repeat(self.padding_mask, self.num_heads, axis=0)
        
        # 编译函数 - 修复：移除partial中的causal参数
        self.flash_attention_jit = jax.jit(partial(
            flash_attention_v1,
            block_size_q=32,
            block_size_kv=32
        ))
    
    def standard_attention(self, q, k, v, padding_mask_k=None, padding_mask_q=None, causal=False):
        """标准注意力实现"""
        # 计算注意力分数
        scores = jnp.einsum('bqd,bkd->bqk', q, k) / jnp.sqrt(k.shape[-1])
        
        # 应用掩码
        if causal:
            mask = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
            scores = jnp.where(mask, scores, -jnp.inf)
        
        if padding_mask_k is not None:
            scores = jnp.where(padding_mask_k[:, None, :], scores, -jnp.inf)
        if padding_mask_q is not None:
            scores = jnp.where(padding_mask_q[:, :, None], scores, -jnp.inf)
        
        # 处理全无效行
        row_valid = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
        scores = jnp.where(row_valid, scores, 0.0)
        
        # 计算注意力权重
        attn_weights = jax.nn.softmax(scores)
        
        # 计算输出
        output = jnp.einsum('bqk,bkd->bqd', attn_weights, v)
        return output
    
    def test_forward_no_mask(self):
        """测试无掩码的前向传播一致性"""
        # FlashAttention 输出
        flash_output = self.flash_attention_jit(
            self.q, self.k, self.v, None, None, False  # 修复：显式传递causal参数
        )
        
        # 标准注意力输出
        standard_output = self.standard_attention(
            self.q, self.k, self.v, None, None, False
        )
        
        # 计算差异
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"无掩码前向传播差异: {diff:.6f}")
        
        # 验证差异在可接受范围内
        self.assertLess(diff, 1e-3, "无掩码前向传播差异过大")
    
    def test_forward_with_mask(self):
        """测试带掩码的前向传播一致性"""
        # FlashAttention 输出
        flash_output = self.flash_attention_jit(
            self.q, self.k, self.v, self.padding_mask, self.padding_mask, False  # 修复：显式传递causal参数
        )
        
        # 标准注意力输出
        standard_output = self.standard_attention(
            self.q, self.k, self.v, self.padding_mask, self.padding_mask, False
        )
        
        # 计算差异
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"带掩码前向传播差异: {diff:.6f}")
        
        # 验证差异在可接受范围内
        self.assertLess(diff, 1e-3, "带掩码前向传播差异过大")
    
    def test_forward_causal_mask(self):
        """测试因果掩码的前向传播一致性"""
        # FlashAttention 输出
        flash_output = self.flash_attention_jit(
            self.q, self.k, self.v, None, None, True  # 修复：显式传递causal参数
        )
        
        # 标准注意力输出
        standard_output = self.standard_attention(
            self.q, self.k, self.v, None, None, True
        )
        
        # 计算差异
        diff = jnp.max(jnp.abs(flash_output - standard_output))
        print(f"因果掩码前向传播差异: {diff:.6f}")
        
        # 验证差异在可接受范围内
        self.assertLess(diff, 1e-3, "因果掩码前向传播差异过大")
    
    def test_backward_no_mask(self):
        """测试无掩码的反向传播一致性"""
        # 定义损失函数
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_jit(q, k, v, None, None, False))  # 修复：显式传递causal参数
        
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(q, k, v, None, None, False))
        
        # 计算梯度
        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)
        
        # 计算梯度差异
        diff_q = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        diff_k = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        diff_v = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        
        print(f"无掩码梯度差异 - Q: {diff_q:.6f}, K: {diff_k:.6f}, V: {diff_v:.6f}")
        
        # 验证差异在可接受范围内
        self.assertLess(diff_q, 1e-3, "Q梯度差异过大")
        self.assertLess(diff_k, 1e-3, "K梯度差异过大")
        self.assertLess(diff_v, 1e-3, "V梯度差异过大")
    
    def test_backward_with_mask(self):
        """测试带掩码的反向传播一致性"""
        # 定义损失函数
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_jit(q, k, v, self.padding_mask, self.padding_mask, False))  # 修复
        
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(q, k, v, self.padding_mask, self.padding_mask, False))
        
        # 计算梯度
        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)
        
        # 计算梯度差异
        diff_q = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        diff_k = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        diff_v = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        
        print(f"带掩码梯度差异 - Q: {diff_q:.6f}, K: {diff_k:.6f}, V: {diff_v:.6f}")
        
        # 验证差异在可接受范围内
        self.assertLess(diff_q, 1e-3, "Q梯度差异过大")
        self.assertLess(diff_k, 1e-3, "K梯度差异过大")
        self.assertLess(diff_v, 1e-3, "V梯度差异过大")
    
    def test_backward_causal_mask(self):
        """测试因果掩码的反向传播一致性"""
        # 定义损失函数
        def loss_flash(q, k, v):
            return jnp.sum(self.flash_attention_jit(q, k, v, None, None, True))  # 修复：显式传递causal参数
        
        def loss_standard(q, k, v):
            return jnp.sum(self.standard_attention(q, k, v, None, None, True))
        
        # 计算梯度
        grad_flash = jax.grad(loss_flash, (0, 1, 2))(self.q, self.k, self.v)
        grad_standard = jax.grad(loss_standard, (0, 1, 2))(self.q, self.k, self.v)
        
        # 计算梯度差异
        diff_q = jnp.max(jnp.abs(grad_flash[0] - grad_standard[0]))
        diff_k = jnp.max(jnp.abs(grad_flash[1] - grad_standard[1]))
        diff_v = jnp.max(jnp.abs(grad_flash[2] - grad_standard[2]))
        
        print(f"因果掩码梯度差异 - Q: {diff_q:.6f}, K: {diff_k:.6f}, V: {diff_v:.6f}")
        
        # 验证差异在可接受范围内
        self.assertLess(diff_q, 1e-3, "Q梯度差异过大")
        self.assertLess(diff_k, 1e-3, "K梯度差异过大")
        self.assertLess(diff_v, 1e-3, "V梯度差异过大")
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 创建极端输入
        q = jnp.full_like(self.q, 1000.0)
        k = jnp.full_like(self.k, -1000.0)
        v = jnp.full_like(self.v, 1e6)
        
        # 计算输出
        output = self.flash_attention_jit(
            q, k, v, self.padding_mask, self.padding_mask, False
        )
        
        # 验证输出值范围
        self.assertTrue(jnp.all(jnp.isfinite(output)))
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isinf(output)))
        print("数值稳定性测试通过")
    
    def test_different_block_sizes(self):
        """测试不同块大小的输出一致性"""
        # 不同块大小配置
        block_sizes = [
            (16, 16),
            (32, 32),
            (64, 64),
            (16, 32),
            (32, 16)
        ]
        
        outputs = []
        
        for block_size_q, block_size_kv in block_sizes:
            output = flash_attention_v1(
                self.q, self.k, self.v, 
                self.padding_mask, self.padding_mask,
                False,
                block_size_q,
                block_size_kv
            )
            outputs.append(output)
        
        # 比较不同块大小的输出
        for i in range(1, len(outputs)):
            diff = jnp.max(jnp.abs(outputs[0] - outputs[i]))
            self.assertLess(diff, 1e-3, f"块大小 {block_sizes[0]} 和 {block_sizes[i]} 的输出差异过大")
        print("不同块大小输出一致性测试通过")

if __name__ == "__main__":
    unittest.main()