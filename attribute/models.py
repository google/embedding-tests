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
from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, alpha=1.0):
    grad_name = "FlipGradient%d" % self.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [tf.negative(grad) * alpha]

    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)

    self.num_calls += 1
    return y


flip_gradient = FlipGradientBuilder()


def build_model(num_attr, hidden_size=0):
  if hidden_size == 0:
    model = tf.keras.Sequential([
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(num_attr, name='output')
    ])
  else:
    model = tf.keras.Sequential([
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu,
                            name='fc'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(num_attr, name='output')
    ])
  model_fn = lambda x, training: model.apply(x, training=training)
  return model_fn


def build_ae_model(num_attr, hidden_size, input_size):
  drop_layer = tf.keras.layers.Dropout(0.25)
  fc_layer = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu,
                                   name='fc')
  output_layer = tf.keras.layers.Dense(num_attr, name='output')
  ae_layer = tf.keras.layers.Dense(input_size, name='ae')

  def model_fn(x, training):
    x = drop_layer(x, training=training)
    x = fc_layer(x)
    x = drop_layer(x, training=training)
    output = output_layer(x)
    ae = ae_layer(x)
    ae = tf.nn.l2_normalize(ae, axis=-1)
    return output, ae

  return model_fn


class TextCNN(object):
  def __init__(self, vocab_size, emb_dim, num_filter=32, init_word_emb=None):
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim

    self.word_emb = tf.Variable(self.get_or_init_word_emb(init_word_emb),
                                name="emb_cnn")
    self.convs = [
      tf.keras.layers.Conv1D(num_filter, kernel_size=3, activation=tf.nn.relu,
                             name='conv1'),
      tf.keras.layers.Conv1D(num_filter, kernel_size=4, activation=tf.nn.relu,
                             name='conv2'),
      tf.keras.layers.Conv1D(num_filter, kernel_size=5, activation=tf.nn.relu,
                             name='conv3')
    ]
    self.dropout = tf.keras.layers.Dropout(0.25)

  def get_or_init_word_emb(self, weights=None):
    if weights is None:
      return tf.random_uniform((self.vocab_size, self.emb_dim), -0.1, 0.1)
    else:
      return tf.convert_to_tensor(weights, dtype=tf.float32)

  def forward(self, inputs, masks, training):
    emb = tf.nn.embedding_lookup(self.word_emb, inputs)
    cnn_inputs = tf.multiply(emb, tf.cast(tf.expand_dims(masks, 2), tf.float32))
    cnn_inputs = self.dropout.apply(cnn_inputs, training=training)
    cnn_outputs = []
    for conv in self.convs:
      cnn_output = conv.apply(cnn_inputs)
      pool_output = tf.reduce_max(cnn_output, axis=1)
      cnn_outputs.append(pool_output)

    cnn_outputs = tf.concat(cnn_outputs, axis=1)
    return cnn_outputs


class TextCharCNN(object):
  def __init__(self, alphabet_size, hidden_size=256, num_filter=32):
    self.alphabet_size = alphabet_size
    self.layers = [
      tf.keras.layers.Conv1D(filters=num_filter, kernel_size=7,
                             activation=tf.nn.relu, name='conv1'),
      tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='pool1'),
      tf.keras.layers.Conv1D(filters=num_filter, kernel_size=7,
                             activation=tf.nn.relu, name='conv2'),
      tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='pool2'),
      tf.keras.layers.Conv1D(filters=num_filter, kernel_size=3,
                             activation=tf.nn.relu, name='conv3'),
      tf.keras.layers.Conv1D(filters=num_filter, kernel_size=3,
                             activation=tf.nn.relu, name='conv4'),
      tf.keras.layers.Conv1D(filters=num_filter, kernel_size=3,
                             activation=tf.nn.relu, name='conv5'),
      tf.keras.layers.Conv1D(filters=num_filter, kernel_size=3,
                             activation=tf.nn.relu, name='conv6'),
      tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='pool6'),
    ]
    self.dropout = tf.keras.layers.Dropout(0.25)
    self.flatten = tf.keras.layers.Flatten()
    self.fc = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu)

  def forward(self, inputs, training):
    inputs = tf.one_hot(inputs, self.alphabet_size)
    for layer in self.layers:
      inputs = layer.apply(inputs)

    inputs = self.flatten(inputs)
    inputs = self.fc(inputs)
    inputs = self.dropout.apply(inputs, training=training)
    return inputs


class MINE(object):
  def __init__(self, hidden_size, num_attr, emb_attr_dim=0):
    self.emb_attr = emb_attr_dim > 0
    if self.emb_attr:
      self.attr_emb = tf.get_variable(
        "attr_emb", shape=(num_attr, emb_attr_dim),
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self.attr_dim = emb_attr_dim
    else:
      self.attr_dim = num_attr

    self.fc1 = tf.keras.layers.Dense(hidden_size,
                                     activation=tf.nn.elu, name='fc1')
    self.fc2 = tf.keras.layers.Dense(hidden_size,
                                     activation=tf.nn.elu, name='fc2')
    self.fc3 = tf.keras.layers.Dense(1, name='fc3')

  def forward_t(self, x, y):
    if self.emb_attr:
      a = tf.nn.embedding_lookup(self.attr_emb, y)
    else:
      a = tf.one_hot(y, self.attr_dim)

    concat = tf.concat([x, a], axis=1)
    fc1 = self.fc1.apply(concat)
    fc2 = self.fc2.apply(fc1)
    fc3 = self.fc3.apply(fc2)
    return fc3

  def forward(self, x, y):
    t_joint = self.forward_t(x, y)
    t_marginal = self.forward_t(x, tf.random.shuffle(y))
    return tf.reduce_mean(t_joint), \
           tf.reduce_logsumexp(t_marginal) \
           - tf.log(tf.cast(tf.shape(x)[0], tf.float32))
