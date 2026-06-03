# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stax is a small but flexible neural net specification library from scratch.

You likely do not mean to import this module! Stax is intended as an example
library only. There are a number of other much more fully-featured neural
network libraries for JAX, including `Flax`_ from Google, and `Haiku`_ from
DeepMind.

.. _Haiku: https://github.com/deepmind/dm-haiku
.. _Flax: https://github.com/google/flax
"""

import functools
import operator as op

import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, standardize)
from jax.nn.initializers import glorot_normal, he_normal, normal, ones, zeros

# aliases for backwards compatibility
glorot = glorot_normal
randn = normal
logsoftmax = log_softmax

# Following the convention used in Keras and tf.layers, we use CamelCase for the
# names of layer constructors, like Conv and Relu, while using snake_case for
# other functions, like lax.conv and relu.

# Each layer constructor function returns an (init_fun, apply_fun) pair, where
#   init_fun: takes an rng key and an input shape and returns an
#     (output_shape, params) pair,
#   apply_fun: takes params, inputs, and an rng key and applies the layer.


def Dense(out_dim, W_init=glorot_normal(), b_init=normal()):
  """Layer constructor function for a dense (fully-connected) layer."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2 = random.split(rng)
    W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
    return output_shape, (W, b)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return jnp.dot(inputs, W) + b
  return init_fun, apply_fun


def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=None,
                b_init=normal(1e-6)):
  """Layer construction function for a general convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or he_normal(rhs_spec.index('I'), rhs_spec.index('O'))
  def init_fun(rng, input_shape):
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [out_chan if c == 'O' else
                    input_shape[lhs_spec.index('C')] if c == 'I' else
                    next(filter_shape_iter) for c in rhs_spec]
    output_shape = lax.conv_general_shape_tuple(
        input_shape, kernel_shape, strides, padding, dimension_numbers)
    bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
    k1, k2 = random.split(rng)
    W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
    return output_shape, (W, b)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                    dimension_numbers=dimension_numbers) + b
  return init_fun, apply_fun
Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))


def GeneralConvTranspose(dimension_numbers, out_chan, filter_shape,
                         strides=None, padding='VALID', W_init=None,
                         b_init=normal(1e-6)):
  """Layer construction function for a general transposed-convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or he_normal(rhs_spec.index('I'), rhs_spec.index('O'))
  def init_fun(rng, input_shape):
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [out_chan if c == 'O' else
                    input_shape[lhs_spec.index('C')] if c == 'I' else
                    next(filter_shape_iter) for c in rhs_spec]
    output_shape = lax.conv_transpose_shape_tuple(
        input_shape, kernel_shape, strides, padding, dimension_numbers)
    bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
    k1, k2 = random.split(rng)
    W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
    return output_shape, (W, b)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return lax.conv_transpose(inputs, W, strides, padding,
                              dimension_numbers=dimension_numbers) + b
  return init_fun, apply_fun
Conv1DTranspose = functools.partial(GeneralConvTranspose, ('NHC', 'HIO', 'NHC'))
ConvTranspose = functools.partial(GeneralConvTranspose,
                                  ('NHWC', 'HWIO', 'NHWC'))


def BatchNorm(axis=(0, 1, 2), epsilon=1e-5, momentum=0.1, center=True, scale=True,
              beta_init=zeros, gamma_init=ones):
  """Batch normalization — JIT-compatible, PyTorch-aligned.

  Defaults match PyTorch ``BatchNorm2d``: ``eps=1e-5``, ``momentum=0.1``.
  Call ``BatchNorm()`` directly — no extra arguments needed.

  ──  QUICK REFERENCE  ──────────────────────────────────────────

  init always returns 3 values; apply always takes 3 positional args
  and returns 2 values::

      # init
      shape, params, bn_state = bn_init(rng, input_shape)
      # params   = (beta, gamma)                     ← learnable
      # bn_state = (running_mean, running_var)        ← mutable

      # training
      (y, new_bn_state) = bn_apply(params, bn_state, x, is_training=True)

      # inference
      (y, _) = bn_apply(params, bn_state, x, is_training=False)

  ──  RULES FOR USERS  ─────────────────────────────────────────

  | 场景                          | 你需要注意的                              |
  |-------------------------------|------------------------------------------|
  | 用现有层 (ConvBN/C3Block 等)  | 什么都不用管，封装好了                     |
  | 模型含 BN，但全用 serial() 搭 | init 多拿一个返回值, apply 多传/多收一个值 |
  | 手写含 BN 的自定义层          | init → 3-tuple, apply → (output, new_state) |
  | 训练循环                      | bn_state 必须作为 train_step 显式参数     |
  | 推理                          | checkout 里必须含 bn_state                |
  |                               | 用 is_training=False, `_` 吞掉第二个返回值 |

  ──  PITFALLS  ─────────────────────────────────────────────────

  1. 推理时只加载了 params 没加载 bn_state → BN 用 rm=0/rv=1 归一化，输出完全错误。
  2. train_step 闭包捕获 bn_state → JIT 按值追踪，每一步都复用第一步的值。
  3. 自定义层 init 返回 2-tuple 但内含 BN → bn_state 丢失，BN 始终用初始值。

  Returns:
    (init_fun, apply_fun) pair.
  """
  _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
  _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()
  axis = (axis,) if jnp.isscalar(axis) else axis

  def init_fun(rng, input_shape):
    shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
    k1, k2 = random.split(rng)
    beta = _beta_init(k1, shape)
    gamma = _gamma_init(k2, shape)
    running_mean = jnp.zeros(shape)
    running_var = jnp.ones(shape)
    return input_shape, (beta, gamma), (running_mean, running_var)

  def apply_fun(params, bn_state, x, **kwargs):
    beta, gamma = params
    running_mean, running_var = bn_state
    is_training = kwargs.get('is_training', True)
    collector = kwargs.get('_bn_collector', None)
    counter = kwargs.get('_bn_counter', None)
    ed = tuple(None if i in axis else slice(None) for i in range(jnp.ndim(x)))

    if is_training or collector is not None:
      mean = jnp.mean(x, axis=axis)
      var = jnp.var(x, axis=axis)
      # EMA update of running statistics (PyTorch-compatible)
      new_rm = (1.0 - momentum) * running_mean + momentum * mean
      new_rv = (1.0 - momentum) * running_var + momentum * var
      new_bn_state = (new_rm, new_rv)
    else:
      mean = running_mean
      var = running_var
      new_bn_state = bn_state

    if collector is not None and counter is not None:
      idx = counter[0]; counter[0] = idx + 1
      collector.setdefault(idx, []).append((mean, var))

    z = (x - mean[ed]) / jnp.sqrt(var[ed] + epsilon)
    if center and scale: return gamma[ed] * z + beta[ed], new_bn_state
    if center: return z + beta[ed], new_bn_state
    if scale: return gamma[ed] * z, new_bn_state
    return z, new_bn_state

  return init_fun, apply_fun


def elementwise(fun, **fun_kwargs):
  """Layer that applies a scalar function elementwise on its inputs."""
  init_fun = lambda rng, input_shape: (input_shape, ())
  apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
  return init_fun, apply_fun
Tanh = elementwise(jnp.tanh)
Relu = elementwise(relu)
Exp = elementwise(jnp.exp)
LogSoftmax = elementwise(log_softmax, axis=-1)
Softmax = elementwise(softmax, axis=-1)
Softplus = elementwise(softplus)
Sigmoid = elementwise(sigmoid)
Elu = elementwise(elu)
LeakyRelu = elementwise(leaky_relu)
Selu = elementwise(selu)
Gelu = elementwise(gelu)


def _pooling_layer(reducer, init_val, rescaler=None):
  def PoolingLayer(window_shape, strides=None, padding='VALID', spec=None):
    """Layer construction function for a pooling layer."""
    strides = strides or (1,) * len(window_shape)
    rescale = rescaler(window_shape, strides, padding) if rescaler else None

    if spec is None:
      non_spatial_axes = 0, len(window_shape) + 1
    else:
      non_spatial_axes = spec.index('N'), spec.index('C')

    for i in sorted(non_spatial_axes):
      window_shape = window_shape[:i] + (1,) + window_shape[i:]
      strides = strides[:i] + (1,) + strides[i:]

    def init_fun(rng, input_shape):
      padding_vals = lax.padtype_to_pads(input_shape, window_shape,
                                         strides, padding)
      ones = (1,) * len(window_shape)
      out_shape = lax.reduce_window_shape_tuple(
        input_shape, window_shape, strides, padding_vals, ones, ones)
      return out_shape, ()
    def apply_fun(params, inputs, **kwargs):
      out = lax.reduce_window(inputs, init_val, reducer, window_shape,
                              strides, padding)
      return rescale(out, inputs, spec) if rescale else out
    return init_fun, apply_fun
  return PoolingLayer
MaxPool = _pooling_layer(lax.max, -jnp.inf)
SumPool = _pooling_layer(lax.add, 0.)


def _normalize_by_window_size(dims, strides, padding):
  def rescale(outputs, inputs, spec):
    if spec is None:
      non_spatial_axes = 0, inputs.ndim - 1
    else:
      non_spatial_axes = spec.index('N'), spec.index('C')

    spatial_shape = tuple(inputs.shape[i]
                          for i in range(inputs.ndim)
                          if i not in non_spatial_axes)
    one = jnp.ones(spatial_shape, dtype=inputs.dtype)
    window_sizes = lax.reduce_window(one, 0., lax.add, dims, strides, padding)
    for i in sorted(non_spatial_axes):
      window_sizes = jnp.expand_dims(window_sizes, i)

    return outputs / window_sizes
  return rescale
AvgPool = _pooling_layer(lax.add, 0., _normalize_by_window_size)


def Flatten():
  """Layer construction function for flattening all but the leading dim."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[0], functools.reduce(op.mul, input_shape[1:], 1)
    return output_shape, ()
  def apply_fun(params, inputs, **kwargs):
    return jnp.reshape(inputs, (inputs.shape[0], -1))
  return init_fun, apply_fun
Flatten = Flatten()


def Identity():
  """Layer construction function for an identity layer."""
  init_fun = lambda rng, input_shape: (input_shape, ())
  apply_fun = lambda params, inputs, **kwargs: inputs
  return init_fun, apply_fun
Identity = Identity()


def FanOut(num):
  """Layer construction function for a fan-out layer."""
  init_fun = lambda rng, input_shape: ([input_shape] * num, ())
  apply_fun = lambda params, inputs, **kwargs: [inputs] * num
  return init_fun, apply_fun


def FanInSum():
  """Layer construction function for a fan-in sum layer."""
  init_fun = lambda rng, input_shape: (input_shape[0], ())
  apply_fun = lambda params, inputs, **kwargs: sum(inputs)
  return init_fun, apply_fun
FanInSum = FanInSum()


def FanInConcat(axis=-1):
  """Layer construction function for a fan-in concatenation layer."""
  def init_fun(rng, input_shape):
    ax = axis % len(input_shape[0])
    # ax = axis % input_shape[0]
    concat_size = sum(shape[ax] for shape in input_shape)
    out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
    return out_shape, ()
  def apply_fun(params, inputs, **kwargs):
    return jnp.concatenate(inputs, axis)
  return init_fun, apply_fun


def Dropout(drop_rate):
    """Dropout 层 —— 训练时随机丢弃，推理时直接透传。

    使用 is_training kwarg 区分训练/推理模式，与 BatchNorm 保持一致。
    训练时会将保留的神经元除以 keep_prob 以保持输出期望不变。

    注意：用 lax.cond 而非 Python if 处理 is_training，因为 is_training
    穿过 @jax.jit 边界后会变成 tracer，Python 无法对其做布尔判断。
    """
    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        is_training = kwargs.get('is_training', True)
        rng = kwargs.get('rng', None)
        keep_prob = 1.0 - drop_rate

        # rng 可能为 None（推理时），此时送入一个占位 key。
        # 该 key 仅在 lax.cond 的 _train 分支中被 trace（不会真实消费），
        # 实际执行时 is_training=False 确保走 _eval 分支。
        safe_rng = rng if rng is not None else jax.random.PRNGKey(0)

        def _train(rng_key):
            keep_mask = random.bernoulli(rng_key, keep_prob, inputs.shape)
            return jnp.where(keep_mask, inputs / keep_prob, 0)

        def _eval(_):
            return inputs

        return lax.cond(is_training, _train, _eval, safe_rng)

    return init_fun, apply_fun


# Composing layers via combinators


def serial(*layers):
  """Combinator for composing layers in serial; threads BN state when present.

  Detects stateful sub-layers by inspecting whether ``init_fun`` returns a
  2-tuple ``(shape, params)`` or a 3-tuple ``(shape, params, state)``.
  Stateful layers receive ``(params, state, x, ...)`` in ``apply_fun`` and
  return ``(output, new_state)``; stateless layers keep the original signature.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new ``(init_fun, apply_fun)`` pair.  ``init_fun`` returns
    ``(output_shape, params, bn_states)`` where *bn_states* is a list
    with ``None`` for stateless sub-layers.  ``apply_fun`` accepts
    ``(params, bn_states, inputs, **kwargs)`` and returns
    ``(output, new_bn_states)``.
  """
  nlayers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(rng, input_shape):
    params = []
    bn_states = []
    for init_fun in init_funs:
      rng, layer_rng = random.split(rng)
      result = init_fun(layer_rng, input_shape)
      if len(result) == 3:
        input_shape, param, bn_st = result
        bn_states.append(bn_st)
      else:
        input_shape, param = result
        bn_states.append(None)
      params.append(param)
    return input_shape, params, bn_states
  def apply_fun(params, bn_states, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
    new_bn_states = []
    for fun, param, bn_st, rng in zip(apply_funs, params, bn_states, rngs):
      if bn_st is not None:
        inputs, new_st = fun(param, bn_st, inputs, rng=rng, **kwargs)
        new_bn_states.append(new_st)
      else:
        inputs = fun(param, inputs, rng=rng, **kwargs)
        new_bn_states.append(None)
    return inputs, new_bn_states
  return init_fun, apply_fun


def parallel(*layers):
  """Combinator for composing layers in parallel; threads BN state.

  Detects stateful sub-layers the same way as :func:`serial`.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new ``(init_fun, apply_fun)`` pair.  ``init_fun`` returns
    ``((output_shapes...), params, bn_states)``.  ``apply_fun`` returns
    ``((outputs...), new_bn_states)``.
  """
  nlayers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(rng, input_shape):
    rngs = random.split(rng, nlayers)
    results = [init(rng, shape) for init, rng, shape
               in zip(init_funs, rngs, input_shape)]
    shapes = [r[0] for r in results]
    params = tuple(r[1] for r in results)
    bn_states = tuple(r[2] if len(r) == 3 else None for r in results)
    return shapes, params, bn_states
  def apply_fun(params, bn_states, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
    outputs = []
    new_bn_states = []
    for f, p, s, x, r in zip(apply_funs, params, bn_states, inputs, rngs):
      if s is not None:
        y, ns = f(p, s, x, rng=r, **kwargs)
        outputs.append(y); new_bn_states.append(ns)
      else:
        outputs.append(f(p, x, rng=r, **kwargs))
        new_bn_states.append(None)
    return outputs, tuple(new_bn_states)
  return init_fun, apply_fun


def shape_dependent(make_layer):
  """Combinator to delay layer constructor pair until input shapes are known.

  Args:
    make_layer: a one-argument function that takes an input shape as an argument
      (a tuple of positive integers) and returns an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the same
    layer as returned by `make_layer` but with its construction delayed until
    input shapes are known.
  """
  def init_fun(rng, input_shape):
    return make_layer(input_shape)[0](rng, input_shape)
  def apply_fun(params, inputs, **kwargs):
    return make_layer(inputs.shape)[1](params, inputs, **kwargs)
  return init_fun, apply_fun

def GroupNorm(num_groups=32, epsilon=1e-5):
    """Correct GroupNorm implementation with actual parameter initialization"""
    def init_fun(rng, input_shape):
        # Input shape: (batch, height, width, channels)
        channels = input_shape[-1]
        
        # Initialize scale (gamma) and bias (beta) parameters
        rng1, rng2 = jax.random.split(rng)
        scale = jax.random.normal(rng1, (1, 1, 1, channels)) * 0.02  # Small random initialization
        bias = jnp.zeros((1, 1, 1, channels))  # Initialize bias to zero
        
        return input_shape, (scale, bias)
    
    def apply_fun(params, inputs, **kwargs):
        scale, bias = params
        batch, height, width, channels = inputs.shape
        
        # Reshape to (batch, height, width, groups, channels_per_group)
        group_size = channels // num_groups
        inputs_reshaped = inputs.reshape((batch, height, width, num_groups, group_size))
        
        # Compute mean and variance per group
        mean = jnp.mean(inputs_reshaped, axis=(1, 2, 4), keepdims=True)
        variance = jnp.var(inputs_reshaped, axis=(1, 2, 4), keepdims=True)
        
        # Normalize
        normalized = (inputs_reshaped - mean) / jnp.sqrt(variance + epsilon)
        normalized = normalized.reshape(inputs.shape)
        
        # Scale and shift
        return normalized * scale + bias
    
    return init_fun, apply_fun
def Lambda(fn, name=None):
    """创建自定义 Lambda 层，支持形状变换"""
    def init_fun(rng, input_shape):
        dummy_input = jnp.zeros(input_shape)
        output = fn(dummy_input)
        return output.shape, ()
    
    def apply_fun(params, inputs, **kwargs):
        return fn(inputs)
    
    return init_fun, apply_fun
def LayerNorm(eps=1e-5):
    """与PyTorch完全一致的层归一化实现"""
    def init_fun(rng, input_shape):
        feature_dim = input_shape[-1]
        gamma = jnp.ones(feature_dim)
        beta = jnp.zeros(feature_dim)
        return input_shape, (gamma, beta)
    

    def apply_fun(params, inputs, **kwargs):
        gamma, beta = params
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)
        inv = 1.0 / jnp.sqrt(variance + eps)
        normalized = (inputs - mean) * inv
        return gamma * normalized + beta
    
    return init_fun, apply_fun

