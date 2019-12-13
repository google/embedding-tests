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


def guess_label(logits_us, temp, binary=False):
  # Compute predicted probability distribution py.
  p_model_y = [tf.nn.sigmoid(logits_u) if binary else tf.nn.softmax(logits_u)
               for logits_u in logits_us]
  p_model_y = tf.reduce_mean(p_model_y, axis=0)
  # Compute the target distribution.
  if not binary:
    p_target = tf.pow(p_model_y, 1. / temp)
    p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
  else:
    p_target = tf.pow(p_model_y, 1. / temp)
    p_target_neg = tf.pow(1 - p_model_y, 1. / temp)
    p_target /= p_target + p_target_neg

  return p_target


class MixMode:
  # A class for mixing data for various combination of labeled and unlabeled.
  # x = labeled example
  # y = unlabeled example
  # For example "xx.yxy" means: mix x with x, mix y with both x and y.
  MODES = 'xx.yy xxy.yxy xx.yxy xx.yx xx. .yy xxy. .yxy .'.split()

  def __init__(self, mode):
    assert mode in self.MODES
    self.mode = mode

  @staticmethod
  def augment_pair(x0, l0, x1, l1, beta, **kwargs):
    del kwargs
    mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x0)[0], 1])
    mix = tf.maximum(mix, 1 - mix)
    index = tf.random_shuffle(tf.range(tf.shape(x0)[0]))
    xs = tf.gather(x1, index)
    ls = tf.gather(l1, index)
    xmix = x0 * mix + xs * (1 - mix)
    lmix = l0 * mix + ls * (1 - mix)
    return xmix, lmix

  @staticmethod
  def augment(x, l, beta, **kwargs):
    return MixMode.augment_pair(x, l, x, l, beta, **kwargs)

  def __call__(self, xl, ll, betal):
    assert len(xl) == len(ll) >= 2
    assert len(betal) == 2
    if self.mode == '.':
      return xl, ll
    elif self.mode == 'xx.':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      return [mx0] + xl[1:], [ml0] + ll[1:]
    elif self.mode == '.yy':
      mx1, ml1 = self.augment(
        tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1])
      return (xl[:1] + tf.split(mx1, len(xl) - 1),
              ll[:1] + tf.split(ml1, len(ll) - 1))
    elif self.mode == 'xx.yy':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = self.augment(
        tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1])
      return ([mx0] + tf.split(mx1, len(xl) - 1),
              [ml0] + tf.split(ml1, len(ll) - 1))
    elif self.mode == 'xxy.':
      mx, ml = self.augment(
        tf.concat(xl, 0), tf.concat(ll, 0),
        sum(betal) / len(betal))
      return (tf.split(mx, len(xl))[:1] + xl[1:],
              tf.split(ml, len(ll))[:1] + ll[1:])
    elif self.mode == '.yxy':
      mx, ml = self.augment(
        tf.concat(xl, 0), tf.concat(ll, 0),
        sum(betal) / len(betal))
      return (xl[:1] + tf.split(mx, len(xl))[1:],
              ll[:1] + tf.split(ml, len(ll))[1:])
    elif self.mode == 'xxy.yxy':
      mx, ml = self.augment(
        tf.concat(xl, 0), tf.concat(ll, 0),
        sum(betal) / len(betal))
      return tf.split(mx, len(xl)), tf.split(ml, len(ll))
    elif self.mode == 'xx.yxy':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = self.augment(tf.concat(xl, 0), tf.concat(ll, 0), betal[1])
      mx1, ml1 = [tf.split(m, len(xl))[1:] for m in (mx1, ml1)]
      return [mx0] + mx1, [ml0] + ml1
    elif self.mode == 'xx.yx':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = zip(*[
        self.augment_pair(xl[i], ll[i], xl[0], ll[0], betal[1])
        for i in range(1, len(xl))
      ])
      return [mx0] + list(mx1), [ml0] + list(ml1)
    raise NotImplementedError(self.mode)


def interleave_offsets(batch_size, nu):
  groups = [batch_size // (nu + 1)] * (nu + 1)
  for x in range(batch_size - sum(groups)):
    groups[-x - 1] += 1
  offsets = [0]
  for g in groups:
    offsets.append(offsets[-1] + g)
  assert offsets[-1] == batch_size
  return offsets


def interleave(xy, batch_size):
  nu = len(xy) - 1
  offsets = interleave_offsets(batch_size, nu)
  xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
  for i in range(1, nu + 1):
    xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
  return [tf.concat(v, axis=0) for v in xy]
