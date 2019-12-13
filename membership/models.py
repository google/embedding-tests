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

import numpy as np
import tensorflow as tf


class OuterMetricModel(object):
  def __init__(self, encoder_dim):
    filters = 16
    kernel_size = (5, 5)
    pool_size = (3, 3)
    self.conv_layers = [
      tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                             activation=tf.nn.tanh, name='conv1'),
      tf.keras.layers.MaxPooling2D(pool_size=pool_size, name='pool1'),
      tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                             activation=tf.nn.tanh, name='conv2'),
      tf.keras.layers.MaxPooling2D(pool_size=pool_size, name='pool2'),
      tf.keras.layers.Flatten(name='flatten')
    ]
    self.fc = tf.keras.layers.Dense(1, name='linear')

  def forward(self, embed_a, embed_b):
    x = tf.matmul(tf.expand_dims(embed_a, 2), tf.expand_dims(embed_b, 1))
    x = tf.expand_dims(x, 3)

    for layer in self.conv_layers:
      x = layer.apply(x)

    logits = self.fc.apply(x)
    return tf.reshape(logits, [-1])


def _ortho_weight(shape, dtype=tf.float32):
  W = np.random.normal(size=shape)
  u, s, v = np.linalg.svd(W)
  return tf.convert_to_tensor(u.astype('float32'), dtype)


class BilinearMetricModel(object):
  def __init__(self, encoder_dim):
    shape = (encoder_dim, encoder_dim)
    self.weight = tf.Variable(initial_value=_ortho_weight(shape),
                              name="bilinear_kernel", shape=shape)
    self.dropout = tf.keras.layers.Dropout(0.2)

  def forward(self, embed_a, embed_b, training):
    embed_a = self.dropout(embed_a, training=training)
    embed_b = self.dropout(embed_b, training=training)
    aw = tf.matmul(embed_a, self.weight)  # B x e
    logits = tf.reduce_sum(tf.multiply(aw, embed_b), axis=1)
    return logits


class LinearMetricModel(object):
  def __init__(self, encoder_dim, tie=False):
    self.fc = tf.keras.layers.Dense(encoder_dim, name='linear',
                                    activation=None)
    self.fc2 = self.fc if tie else tf.keras.layers.Dense(
        encoder_dim, name='linear2', activation=None)

    self.dropout = tf.keras.layers.Dropout(0.2)

  def forward(self, embed_a, embed_b, training):
    a = self.fc(embed_a)
    a = self.dropout(a, training=training)

    b = self.fc2(embed_b)
    b = self.dropout(b, training=training)

    logits = tf.reduce_sum(tf.multiply(a, b), axis=1)
    return tf.reshape(logits, [-1])


class DeepSetModel(object):
  def __init__(self, set_dim):
    self.upsample = tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                                           strides=1, name='upsample')

    self.phi = tf.keras.models.Sequential([
      tf.keras.layers.Dense(set_dim, name='phi1', activation=tf.nn.elu),
      tf.keras.layers.Dense(set_dim, name='phi2', activation=None),
    ])

    self.rho = tf.keras.models.Sequential([
      tf.keras.layers.Dense(set_dim, name='rho1', activation=tf.nn.elu),
      tf.keras.layers.Dense(1, name='rho2', activation=None),
    ])
    self.dropout = tf.keras.layers.Dropout(0.2)

  def forward(self, embed_a, embed_b, training):
    x = tf.multiply(embed_a, embed_b)
    x = tf.expand_dims(x, 2)
    x = self.upsample(x)
    x = self.dropout(x, training=training)

    phi_x = self.phi(x)
    phi_x = self.dropout(phi_x, training=training)

    phi_x = tf.reduce_sum(phi_x, axis=1)
    logits = self.rho(phi_x)
    return tf.reshape(logits, [-1])


class RBFMetricModel(object):
  def __init__(self, encoder_dim):
    if encoder_dim > 0:
      self.fc = tf.keras.layers.Dense(encoder_dim, name='linear',
                                      use_bias=False)
    else:
      self.fc = lambda x: x

  def forward(self, embed_a, embed_b):
    a = self.fc(embed_a)
    b = self.fc(embed_b)
    logits = tf.reduce_sum(tf.multiply(a, b), axis=1)
    return tf.reshape(logits, [-1])


class CovMetricModel(object):
  def __init__(self, encoder_dim):
    self.fc = tf.keras.layers.Dense(encoder_dim // 2, name='linear',
                                    use_bias=False)

    self.encoder_dim = encoder_dim // 2

  def forward(self, embed_a, embed_b):
    a = self.fc.apply(embed_a)
    b = self.fc.apply(embed_b)
    covar = tf.matmul(
      tf.expand_dims(a - tf.reduce_mean(a, axis=1, keepdims=True), 2),
      tf.expand_dims(b - tf.reduce_mean(b, axis=1, keepdims=True), 1))

    covar /= tf.constant(self.encoder_dim, dtype=tf.float32)
    logits = tf.norm(covar, ord='fro', axis=[-2, -1])
    return logits


class DistanceMetricModel(object):
  def __init__(self, encoder_dim):
    self.fc = tf.keras.layers.Dense(encoder_dim // 2, name='linear')

  def forward(self, embed_a, embed_b):
    a = self.fc.apply(embed_a)
    b = self.fc.apply(embed_b)
    logits = - tf.norm(a - b, axis=1)
    return tf.reshape(logits, [-1])
