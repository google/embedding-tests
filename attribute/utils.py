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
import numpy as np
import tqdm

from collections import Counter, defaultdict


def add_gaussian_noise(inputs, stddev=1.0, gamma=1.0, norm=True):
  noise = tf.random_normal(shape=tf.shape(inputs), mean=0., stddev=stddev)
  if norm:
    noise = tf.nn.l2_normalize(noise, axis=-1)
  return inputs + gamma * noise


def batch_interpolation(inputs, alpha=0.75, random=True):
  b = tf.shape(inputs)[0]
  if random:
    index = tf.random_shuffle(tf.range(b))
  else:
    diff = tf.expand_dims(inputs, 1) - tf.expand_dims(inputs, 0)
    l2_distance = tf.reduce_sum(diff ** 2, axis=-1)
    l2_distance = tf.matrix_set_diag(l2_distance, tf.ones(b) * 1e7)
    index = tf.argmin(l2_distance, axis=-1)

  indexed_inputs = tf.gather(inputs, index)
  return alpha * inputs + (1 - alpha) * indexed_inputs


def acc_metrics(logits, labels, num_attr):
  predictions = tf.argmax(logits, axis=-1)
  correct = tf.cast(tf.equal(predictions, labels), tf.float32)
  accuracies = tf.reduce_sum(correct)

  if num_attr > 100:
    top5_correct = tf.nn.in_top_k(logits, labels, 5)
    top5_accuracies = tf.reduce_sum(tf.cast(top5_correct, tf.float32))
  else:
    top5_accuracies = tf.constant(0.)
  return accuracies, top5_accuracies, predictions


def random_replace_words(sents, masks, rate=0.25, vocab_size=50001):
  aug_sents = np.copy(sents)
  sampled = np.arange(1, vocab_size)
  np.random.shuffle(sampled)
  sampled_ptr = len(sampled)
  n_words = np.sum(masks)
  random_probs = np.random.rand(n_words)
  random_probs_ptr = n_words

  for i in range(len(sents)):
    length = sum(masks[i])

    for j in range(length):
      random_probs_ptr -= 1
      if random_probs[random_probs_ptr] <= rate:
        sampled_ptr -= 1
        aug_sents[i, j] = sampled[sampled_ptr]

  return aug_sents


def accuracy_per_class(predict_label, true_label, classes):
  if isinstance(classes, int):
    nclass = classes
    classes = range(nclass)
  else:
    nclass = len(classes)

  acc_per_class = []
  for i in range(nclass):
    idx = true_label == classes[i]
    if idx.sum() != 0:
      acc_per_class.append(
        np.sum(true_label[idx] == predict_label[idx]) / float(idx.sum()))

  if len(acc_per_class) == 0:
    return 0.

  return np.array(acc_per_class).mean()


def get_attrs_to_ids(all_attrs):
  unique_attrs = np.sort(np.unique(all_attrs))
  attribute_to_ids = dict(zip(unique_attrs, np.arange(len(unique_attrs))))
  return attribute_to_ids


def filter_infrequent_attribute(y, thresh):
  count_attr = Counter(y)
  kept_indices = []
  for i, a in enumerate(y):
    if count_attr[a] >= thresh:
      kept_indices.append(i)

  kept_indices = np.asarray(kept_indices)
  kept_y = y[kept_indices]
  print('Filtered {} attribute with {} data below {}'.format(
      len(count_attr) - len(np.unique(kept_y)),
      len(y) - len(kept_indices), thresh))
  return kept_indices


def train_test_split_by_attribute(y, train_size, random_state=12345,
                                  top_k=None, target_test_size=None):
  np.random.seed(random_state)
  n = len(y)

  if top_k > 0:
    count_y = Counter(y)
    top_y = count_y.most_common(top_k)
    unique_attributes = [tup[0] for tup in top_y]
    n = sum(tup[1] for tup in top_y)
  else:
    unique_attributes = np.unique(y)

  train_indices, test_indices = [], []

  test_size = n - train_size * len(unique_attributes)
  valid_test_size = test_size // len(unique_attributes)

  a_indices_dict = defaultdict(list)
  for i, a in enumerate(y):
    a_indices_dict[a].append(i)

  for attribute in tqdm.tqdm(unique_attributes):
    a_indices = a_indices_dict[attribute]

    if target_test_size is not None:
      assert len(a_indices) > 2 * target_test_size
      test_a_indices = np.random.choice(a_indices, 2 * target_test_size,
                                        replace=False)
    else:
      test_a_indices = np.random.choice(a_indices, valid_test_size,
                                        replace=False)

    train_a_indices = np.setdiff1d(a_indices, test_a_indices)
    assert len(train_a_indices) >= train_size
    train_a_indices = np.random.choice(train_a_indices,
                                       train_size, replace=False)

    train_indices.append(train_a_indices)
    test_indices.append(test_a_indices)

  train_indices = np.concatenate(train_indices)
  test_indices = np.concatenate(test_indices)

  assert len(np.intersect1d(train_indices, test_indices)) == 0
  np.random.seed(None)

  return train_indices, test_indices


def tp_fp_fn_metric(y_true, y_pred, num_attr):
  y_true = tf.cast(tf.one_hot(y_true, num_attr), tf.bool)
  y_pred = tf.cast(tf.one_hot(y_pred, num_attr), tf.bool)

  tp = tf.count_nonzero(tf.logical_and(y_pred, y_true), axis=None)
  fp = tf.count_nonzero(tf.logical_and(y_pred, tf.logical_not(y_true)),
                        axis=None)
  fn = tf.count_nonzero(tf.logical_and(tf.logical_not(y_pred), y_true),
                        axis=None)
  return tp, fp, fn
