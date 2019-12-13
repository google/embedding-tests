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


def sents_to_labels(sents, label_thresh=0):
  labels = []
  for s in sents:
    label = s[s > label_thresh]
    assert len(label) > 0, s
    labels.append(label)
  return np.asarray(labels)


def filter_sent_labels(sents, masks, prefix_len=3, max_len=5):
  labels = []
  prefixes = []
  indices = []

  for idx, label in enumerate(sents):
    length = np.sum(masks[idx])
    label = label[:length]
    if np.any(label == 0):
      continue

    label_prefix = label[:prefix_len]
    label_to_predict = label[prefix_len:]

    if len(label_to_predict) == max_len:
      prefixes.append(label_prefix)
      labels.append(label_to_predict)
      indices.append(idx)

  return np.asarray(indices), np.asarray(prefixes), np.asarray(labels)


def filter_unknown_sents(sents, masks, unk_ratio=0.5, seq_len=None,
                         min_len=5, max_len=None):
  labels = []
  indices = []

  for idx, label in enumerate(sents):
    length = np.sum(masks[idx])
    if length < min_len:
      continue

    label = label[:length]
    if np.sum(label == 0) >= max(int(length * unk_ratio), 1):
      continue

    if seq_len is not None and length != seq_len:
      continue

    if max_len is not None and length > max_len:
      continue

    labels.append(label)
    indices.append(idx)

  return np.asarray(indices), np.asarray(labels)


def tp_fp_fn_metrics(y_true, y_pred):
  y_true = tf.cast(y_true, tf.bool)
  y_pred = tf.cast(y_pred, tf.bool)

  tp = tf.count_nonzero(tf.logical_and(y_pred, y_true), axis=None)
  fp = tf.count_nonzero(tf.logical_and(y_pred, tf.logical_not(y_true)),
                        axis=None)
  fn = tf.count_nonzero(tf.logical_and(tf.logical_not(y_pred), y_true),
                        axis=None)
  return tp, fp, fn


def count_label_freq(labels, num_words):
  freq = np.ones(num_words)
  for label in labels:
    freq[label] += 1
  return freq


def multi_hot(ind, depth):
  one_hots = tf.vectorized_map(lambda i: tf.one_hot(i, depth),
                               tf.transpose(ind))
  multi_hots = tf.reduce_any(tf.cast(one_hots, tf.bool), axis=0)
  return tf.cast(multi_hots, tf.float32)


def tp_fp_fn_metrics_np(y_pred, y_true, rtn_match=False, rare_thresh=0):
  tp, fp, fn = 0., 0., 0.
  matched = {}
  i = 0
  for p, t in zip(y_pred, y_true):
    t = np.unique(t[t > 0])
    p = np.unique(p[p > 0])
    matched_all = np.intersect1d(t, p)

    if rtn_match:
      matched_rare = np.intersect1d(t[t > rare_thresh], p[p > rare_thresh])
      if len(matched_rare) >= 1:
        matched[i] = matched_all

    tp += len(matched_all)  # predicted and in label
    fp += len(np.setdiff1d(p, t))  # predicted but not in label
    fn += len(np.setdiff1d(t, p))  # in label but not predicted
    i += 1

  if rtn_match:
    return tp, fp, fn, matched

  return tp, fp, fn


def exact_match(y_pred, y_true):
  assert y_pred.shape == y_true.shape
  equal = np.all(y_pred == y_true, axis=1)
  matched = np.arange(len(y_pred))[equal]
  return matched


def sinkhorn(log_alpha, n_iters=5):
  n = tf.shape(log_alpha)[1]
  log_alpha = tf.reshape(log_alpha, [-1, n, n])

  for _ in range(n_iters):
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2), [-1, n, 1])
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1), [-1, 1, n])
  return tf.exp(log_alpha)


def continuous_topk(x, k, t=1.0, epsilon=1e-7, unroll=True):
  if unroll:
    logits_list = []
    onehot_approx_i = tf.zeros_like(x)
    x_i = x
    for _ in range(k):
      khot_mask = tf.maximum(1.0 - onehot_approx_i, epsilon)
      x_i += tf.log(khot_mask)
      onehot_approx_i = tf.nn.softmax(x_i / t, axis=-1)
      logits_list.append(tf.expand_dims(x_i, axis=1))
    logits_list = tf.concat(logits_list, axis=1)
  else:
    b, e = x.shape.as_list()
    logits_list = tf.zeros((b, 0, e))

    def cond_fn(i, x_i, onehot_approx_i, logits_list_i):
      return tf.less(i, k)

    def body_fn(i, x_i, onehot_approx_i, logits_list_i):
      khot_mask = tf.maximum(1.0 - onehot_approx_i, epsilon)
      x_i += tf.log(khot_mask)
      onehot_approx_i = tf.nn.softmax(x_i / t, axis=-1)
      logits_list_i = tf.concat([logits_list_i, tf.expand_dims(x_i, 1)], 1)
      return i + 1, x_i, onehot_approx_i, logits_list_i

    _, _, _, khot_list = tf.while_loop(
      cond_fn, body_fn,
      [tf.constant(0), x, tf.zeros_like(x), logits_list],
      [tf.TensorShape([]), x.get_shape(), x.get_shape(),
       tf.TensorShape([b, None, e])]
    )
  return logits_list


def continuous_topk_v2(x, k, t=1.0, inf=1e7):
  prob_lists = []
  # x_i = x
  v = x.shape.as_list()[-1]
  onehot_i = tf.zeros((x.shape.as_list()[0], v))

  for i in range(k):
    x_i = x[:, i] + onehot_i * (-inf)
    onehot_approx_i = tf.nn.softmax(x_i / t, axis=-1)
    onehot_i = tf.one_hot(tf.argmax(onehot_approx_i, axis=-1), v)
    # x_i = x_i + onehot_i * (-inf)
    prob_lists.append(tf.expand_dims(onehot_approx_i, axis=1))
  return tf.concat(prob_lists, axis=1)
