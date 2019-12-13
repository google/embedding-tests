# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_NEG_INF_FP32 = -1e9


class Attention(object):
  def __init__(self, hidden_size, num_heads=1, attention_dropout=0.,
               train=True, name_prefix=""):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=False,
                                               name=name_prefix + "q")

    self.k_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=False,
                                               name=name_prefix + "k")

    self.v_dense_layer = tf.keras.layers.Dense(hidden_size, use_bias=False,
                                               name=name_prefix + "v")

    self.output_dense_layer = tf.keras.layers.Dense(
        hidden_size, use_bias=False, name=name_prefix + "output_transform")

  def split_heads(self, x):
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def forward(self, x, y, bias, cache=None):
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if self.train:
      weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output, weights


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def forward(self, x, bias, cache=None):
    return super(SelfAttention, self).forward(x, x, bias, cache)


class FeedForwardNetwork(object):
  def __init__(self, hidden_size, filter_size, relu_dropout, train,
               name_prefix=""):
    super(FeedForwardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train

    self.filter_dense_layer = tf.keras.layers.Dense(
      filter_size, use_bias=True, activation=tf.nn.relu,
      name=name_prefix + "filter_layer")

    self.output_dense_layer = tf.keras.layers.Dense(
      hidden_size, use_bias=True, name=name_prefix + "output_layer")

  def forward(self, x, padding=None):
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"):
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1])

        nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, self.hidden_size])
        x = tf.gather_nd(x, indices=nonpad_ids)

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, self.hidden_size])
        x = tf.expand_dims(x, axis=0)

    output = self.filter_dense_layer(x)
    if self.train:
      output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
    output = self.output_dense_layer(output)

    if padding is not None:
      with tf.name_scope("re_add_padding"):
        output = tf.squeeze(output, axis=0)
        output = tf.scatter_nd(
            indices=nonpad_ids,
            updates=output,
            shape=[batch_size * length, self.hidden_size]
        )
        output = tf.reshape(output, [batch_size, length, self.hidden_size])
    return output


class LayerNormalization(object):
  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())

  def forward(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
  def __init__(self, layer, hidden_size, train, layer_postprocess_dropout=0.):
    self.layer = layer
    self.train = train
    self.postprocess_dropout = layer_postprocess_dropout
    # Create normalization layer
    self.layer_norm = LayerNormalization(hidden_size)

  def forward(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm.forward(x)
    # Get layer output
    y = self.layer.forward(y, *args, **kwargs)
    if isinstance(self.layer, SelfAttention):
      y, a = y
    else:
      a = None
    # Postprocessing: apply dropout and residual connection
    if self.train:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)

    if a is None:
      return x + y
    else:
      return x + y, a


class TransformerBlock(object):
  def __init__(self, hidden_size, filter_size, num_heads=8,
               attention_dropout=0.1, relu_dropout=0.1,
               layer_postprocess_dropout=0.1, num_layer=2,
               train=True, name_prefix=""):
    self.layers = []
    self.outputs = []
    self.attention_matrices = []

    for i in range(num_layer):
      with tf.variable_scope("self_attention_{}".format(i)):
        attn = SelfAttention(hidden_size, num_heads, attention_dropout, train,
                             name_prefix)
        attn = PrePostProcessingWrapper(attn, hidden_size,
                                        layer_postprocess_dropout)
      with tf.variable_scope("ffn_{}".format(i)):
        ffn = FeedForwardNetwork(hidden_size, filter_size, relu_dropout, train,
                                 name_prefix)
        ffn = PrePostProcessingWrapper(ffn, hidden_size,
                                       layer_postprocess_dropout)
      self.layers.append([attn, ffn])

    with tf.variable_scope("output_norm"):
      self.output_normalization = LayerNormalization(hidden_size)

  def forward(self, inputs, masks, pool=True):
    batch_size = tf.shape(inputs)[0]
    length = tf.shape(inputs)[1]

    attention_bias = get_padding_bias(masks)
    inputs_padding = get_padding(masks)
    encoder_inputs = inputs
    masks = tf.cast(masks, tf.float32)

    def mean_pool(oo):
      oo = oo * tf.reshape(masks, [batch_size, length, 1])
      ss = tf.reduce_sum(oo, axis=1)
      ll = tf.reduce_sum(masks, 1, keepdims=True)
      return ss / ll

    for layer in self.layers:
      attn, ffn = layer
      encoder_inputs, a = attn.forward(encoder_inputs, attention_bias, None)
      self.outputs.append(mean_pool(encoder_inputs))
      self.attention_matrices.append(a)
      encoder_inputs = ffn.forward(encoder_inputs, inputs_padding)

    outputs = self.output_normalization.forward(encoder_inputs)
    if pool:
      outputs = mean_pool(outputs)

    return outputs


def get_padding_bias(x):
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF_FP32
    attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1),
                                    axis=1)
  return attention_bias


def get_padding(x, padding_value=0, dtype=tf.float32):
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_value), dtype)
