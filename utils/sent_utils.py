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


from collections import defaultdict
import tqdm
import tensorflow as tf
import numpy as np
import os


def count_rareness(train_sents, train_masks, test_sents, test_masks,
                   percentile=75, verbose=False):
  train_sents = np.vstack(train_sents)
  train_masks = np.vstack(train_masks)
  test_sents = np.vstack(test_sents)
  test_masks = np.vstack(test_masks)

  train_rs = np.sum(train_sents, axis=1) / np.sum(train_masks, axis=1)
  test_rs = np.sum(test_sents, axis=1) / np.sum(test_masks, axis=1)
  if verbose:
    print(np.mean(train_rs), np.mean(test_rs))
    print(np.median(train_rs), np.median(test_rs))

  return np.percentile(train_rs, percentile)


def make_parallel(fn, num_gpus, **kwargs):
  in_splits = {}
  for k, v in kwargs.items():
    in_splits[k] = tf.split(v, num_gpus)

  out_split = []
  for i in range(num_gpus):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        out_split.append(fn(**{k: v[i] for k, v in in_splits.items()}))

  return tf.concat(out_split, 0)


def group_indices_by_len(texts, bs=20):
  # Bucket samples by source sentence length
  bucket_indices = defaultdict(list)

  batches = []
  for i, t in enumerate(texts):
    bucket_indices[len(t)].append(i)

  for l, bucket in bucket_indices.items():
    num_batches = int(np.ceil(len(bucket) * 1.0 / bs))
    for i in range(num_batches):
      cur_batch_size = bs if i < num_batches - 1 else len(bucket) - bs * i
      batches.append([bucket[i * bs + j] for j in range(cur_batch_size)])
  return batches


def group_texts_by_len(texts, bs=20):
  # Bucket samples by source sentence length
  buckets = defaultdict(list)
  bucket_indices = defaultdict(list)

  batches = []
  batch_indices = []
  for i, t in enumerate(texts):
    buckets[len(t)].append(t)
    bucket_indices[len(t)].append(i)

  for l, bucket in buckets.items():
    num_batches = int(np.ceil(len(bucket) * 1.0 / bs))
    for i in range(num_batches):
      cur_batch_size = bs if i < num_batches - 1 else len(bucket) - bs * i
      batches.append([bucket[i * bs + j] for j in range(cur_batch_size)])
      batch_indices.append([bucket_indices[l][i * bs + j]
                            for j in range(cur_batch_size)])

  return batches, batch_indices


def iterate_minibatches_indices(n, batch_size, shuffle=False,
                                include_last=True):
  indices = np.arange(n)

  num_batches = n // batch_size
  if include_last:
    num_batches += (n % batch_size != 0)

  if num_batches == 0:
    num_batches = 1

  if shuffle:
    np.random.shuffle(indices)

  for batch_idx in range(num_batches):
    batch_indices = indices[batch_idx * batch_size:
                            (batch_idx + 1) * batch_size]
    yield batch_indices


def inf_batch_iterator(n, batch_size, shuffle=True, include_last=False):
  while True:
    for indices in iterate_minibatches_indices(n, batch_size,
                                               shuffle, include_last):
      yield indices


def load_embeds(filenames, emb_dir, rtn_filenames=False, stack=True):
  embeds = []
  fnames = []

  for filename in tqdm.tqdm(filenames):
    data_path = os.path.join(emb_dir, filename + '.npz')
    if os.path.exists(data_path):
      with np.load(data_path) as f:
        embed = f['arr_0']
      fnames.append(filename)
      embeds.append(embed)
  n_files = len(fnames)
  if stack:
    fnames = [[fname] * len(emb) for emb, fname in zip(embeds, fnames)]
    fnames = np.concatenate(fnames)
    embeds = np.vstack(embeds)

  print('Loaded {} sets of embeddings from {} files'.format(
    len(embeds), n_files))

  if rtn_filenames:
    return fnames, embeds

  return embeds


def load_raw_sents(filenames, feat_dir, rtn_filenames=False, stack=True):
  sents = []
  masks = []
  fnames = []

  for filename in tqdm.tqdm(filenames):
    data_path = os.path.join(feat_dir, filename + '.npz')
    if os.path.exists(data_path):
      with np.load(data_path) as f:
        fnames.append(filename)
        sents.append(f['arr_0'])
        masks.append(f['arr_1'])

  n_files = len(fnames)
  if stack:
    fnames = [[fname] * len(emb) for emb, fname in zip(sents, fnames)]
    fnames = np.concatenate(fnames)
    sents = np.vstack(sents)
    masks = np.vstack(masks)

  print('Loaded {} data from {} files'.format(
    len(sents), n_files))

  if rtn_filenames:
    return fnames, sents, masks

  return sents, masks


def get_similarity_metric(x, y, metric, rtn_loss=False):
  if metric == 'cosine':
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)
    sim = tf.reduce_sum(tf.multiply(x, y), axis=1)
    loss = 1 - sim
  elif metric == 'log':
    dot = tf.reduce_sum(tf.multiply(x, y), axis=1)
    sim = tf.log_sigmoid(dot)
    loss = -sim
  elif metric == 'dot':
    sim = tf.reduce_sum(tf.multiply(x, y), axis=1)
    loss = -sim
  elif metric == 'l2':
    # sim = -tf.reduce_sum(tf.square(x - y), axis=-1)
    sim = -tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1))
    loss = -sim   # / tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
  else:
    raise ValueError('No such metric', metric)
  return loss if rtn_loss else sim
