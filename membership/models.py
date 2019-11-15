from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


class BilinearMetricModel(object):
  def __init__(self, encoder_dim):
    self.weight = tf.get_variable(
      "bilinear_w", shape=(encoder_dim, encoder_dim))

  def forward(self, embed_a, embed_b):
    aw = tf.matmul(embed_a, self.weight)  # B x e
    logits = tf.reduce_sum(tf.multiply(aw, embed_b), axis=1)
    return logits


class LinearMetricModel(object):
  def __init__(self, encoder_dim):
    self.fc = tf.keras.layers.Dense(encoder_dim, name='linear',
                                    use_bias=False)

  def forward(self, embed_a, embed_b):
    a = self.fc.apply(embed_a)
    b = self.fc.apply(embed_b)
    logits = tf.reduce_sum(tf.multiply(a, b), axis=1)
    return tf.reshape(logits, [-1])


class DistanceMetricModel(object):
  def __init__(self, encoder_dim):
    self.fc = tf.keras.layers.Dense(encoder_dim // 2, name='linear')

  def forward(self, embed_a, embed_b):
    a = self.fc.apply(embed_a)
    b = self.fc.apply(embed_b)
    logits = - tf.norm(a - b, axis=1)
    return tf.reshape(logits, [-1])
