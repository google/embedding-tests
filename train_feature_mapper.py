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

from utils.sent_utils import inf_batch_iterator, iterate_minibatches_indices
from utils.common_utils import log

import sklearn.linear_model as linear_model
import tensorflow as tf
import numpy as np


def linear_mapping(x, y, n_jobs=8):
  # train least square
  projection = linear_model.LinearRegression(n_jobs=n_jobs)
  projection.fit(x, y)
  return projection.predict


def mlp_mapping(x, y, epochs=30, batch_size=256, lr=1e-3, wd=1e-4,
                activation=tf.nn.relu, norm=False):
  dim = y.shape[1]
  l2_reg = tf.keras.regularizers.l2(wd)
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(dim, activation=activation, name='reg_fc',
                                  kernel_regularizer=l2_reg))
  model.add(tf.keras.layers.Dense(dim, name='reg_output',
                                  kernel_regularizer=l2_reg))
  if norm:
    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1),
                                     name='reg_norm'))
  model.compile(tf.keras.optimizers.Adam(lr), loss='mse', metric='mse')
  model.fit(x, y, batch_size=batch_size, verbose=1, epochs=epochs)

  def mapping(z):
    mapped = model.predict(z, batch_size=1024)
    tf.keras.backend.clear_session()
    return mapped

  return mapping


class Generator(object):
  def __init__(self, output_dim, activation=tf.nn.relu):
    self.fc1 = tf.keras.layers.Dense(output_dim, activation=activation,
                                     name='gen_fc1')
    # self.fc2 = tf.keras.layers.Dense(output_dim, activation=activation,
    #                                  name='gen_fc2')
    self.output = tf.keras.layers.Dense(output_dim, name='gen_output')

  def forward(self, x):
    x = self.fc1(x)
    # x = self.fc2(x)
    x = self.output(x)
    return x


class Discriminator(object):
  def __init__(self, output_dim, activation=tf.nn.relu):
    self.fc1 = tf.keras.layers.Dense(output_dim, activation=activation,
                                     name='disc_fc1')
    # self.fc2 = tf.keras.layers.Dense(output_dim, activation=activation,
    #                                  name='disc_fc2')
    self.output = tf.keras.layers.Dense(1, name='disc_output')

  def forward(self, y):
    y = self.fc1(y)
    # y = self.fc2(y)
    y = self.output(y)
    return y


class WGANGP(object):
  """Learn a GAN that translates x to y"""
  def __init__(self, x_dim, y_dim, lr=5e-4, lmbda=10., gamma=0., beta1=0.5,
               activation=tf.nn.relu):
    self.generator = Generator(y_dim, activation)
    self.discriminator = Discriminator(y_dim, activation)
    self.input_x = tf.placeholder(tf.float32, shape=(None, x_dim), name='x')
    self.input_y = tf.placeholder(tf.float32, shape=(None, y_dim), name='y')

    self.real_score = self.discriminator.forward(self.input_y)
    self.fake_y = self.generator.forward(self.input_x)
    self.fake_score = self.discriminator.forward(self.fake_y)

    # calculate gradient penalty
    alpha = tf.random.uniform(shape=tf.shape(self.fake_y), minval=0.,
                              maxval=1.)
    self.interpolate_y = alpha * self.input_y + self.fake_y
    grad = tf.gradients(self.discriminator.forward(self.interpolate_y),
                        [self.interpolate_y])[0]
    grad_norm = tf.norm(grad, axis=1, ord='euclidean')
    self.grad_pen = tf.reduce_mean(tf.square(grad_norm - 1))

    t_vars = tf.trainable_variables()

    self.disc_params = [v for v in t_vars if v.name.startswith('disc')]
    self.disc_optimizer = tf.train.AdamOptimizer(lr, beta1=beta1)
    self.disc_loss = tf.reduce_mean(self.fake_score) - tf.reduce_mean(
      self.real_score)
    self.disc_train_ops = self.disc_optimizer.minimize(
        self.disc_loss + lmbda * self.grad_pen, var_list=self.disc_params)

    l2_loss = tf.reduce_sum(tf.square(self.fake_y - self.input_y), axis=-1)
    self.l2_loss = tf.reduce_mean(l2_loss)

    self.gen_params = [v for v in t_vars if v.name.startswith('gen')]
    self.gen_optimizer = tf.train.AdamOptimizer(lr, beta1=beta1)
    self.gen_loss = -tf.reduce_mean(self.fake_score)
    self.gen_train_ops = self.gen_optimizer.minimize(
        self.gen_loss + gamma * self.l2_loss, var_list=self.gen_params)

  def generate(self, sess, x):
    return sess.run(self.fake_y, {self.input_x: x})

  def train_disc_one_batch(self, sess, x, y):
    d_err, _ = sess.run([self.disc_loss, self.disc_train_ops],
                        {self.input_x: x, self.input_y: y})
    return d_err

  def train_gen_one_batch(self, sess, x, y):
    g_err, l2_err, _ = sess.run([-self.gen_loss, self.l2_loss,
                                 self.gen_train_ops],
                                {self.input_x: x, self.input_y: y})
    return g_err, l2_err


def gan_mapping(x, y, lr=1e-4, lmbda=10., gamma=0., beta1=0.5,
                activation=tf.nn.relu, epoch=30, disc_iters=10,
                batch_size=128):
  n_data, x_dim = x.shape
  y_dim = y.shape[1]
  model = WGANGP(x_dim, y_dim, lr=lr, lmbda=lmbda, gamma=gamma, beta1=beta1,
                 activation=activation)

  gen_sampler = inf_batch_iterator(n_data, batch_size)
  disc_sampler = inf_batch_iterator(n_data, batch_size)
  num_batch_per_epoch = n_data // batch_size + (n_data % batch_size) != 0

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  for e in range(epoch):
    train_d_loss = []
    train_g_loss = []
    train_l2_loss = []
    for _ in range(num_batch_per_epoch):
      # train disc first
      for _ in range(disc_iters):
        disc_idx = next(disc_sampler)
        disc_x, disc_y = x[disc_idx], y[disc_idx]
        d_err = model.train_disc_one_batch(sess, disc_x, disc_y)
        train_d_loss.append(d_err)

      gen_idx = next(gen_sampler)
      gen_x, gen_y = x[gen_idx], y[gen_idx]
      g_err, l2_err = model.train_gen_one_batch(sess, gen_x, gen_y)
      train_g_loss.append(g_err)
      train_l2_loss.append(l2_err)

    train_d_loss = np.mean(train_d_loss)
    train_g_loss = np.mean(train_g_loss)
    train_l2_loss = np.mean(train_l2_loss)
    log('Epoch: {}, disc loss: {:.4f}, gen loss: {:.4f},'
        ' l2 loss: {:.4f}'.format(
          e + 1, train_d_loss, train_g_loss, train_l2_loss))

  def mapping(z):
    mapped = []
    for idx in iterate_minibatches_indices(len(z), batch_size=2048):
      batch_mapped = model.generate(sess, z[idx])
      mapped.append(batch_mapped)

    tf.keras.backend.clear_session()
    return np.vstack(mapped)

  return mapping
