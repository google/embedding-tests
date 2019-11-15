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

from __future__ import division
from __future__ import print_function

import os
import numpy as np

from glove import Glove
from gensim.models import Word2Vec
from common_utils import load_trained_variable


def get_ranks(scores):
  sorter = np.argsort(scores)
  n = len(sorter)
  inv = np.empty(n, dtype=np.intp)
  inv[sorter] = np.arange(n, dtype=np.intp) + 1
  return inv


def get_ranks_from_preds(targets, logits):
  targets_shape = targets.shape

  if len(targets_shape) == 2:
    targets = targets.flatten()
    logits = logits.reshape(targets_shape[0] * targets_shape[1], -1)

  arr = np.asarray(logits)
  algo = 'mergesort'
  sorter = np.argsort(-arr, axis=-1, kind=algo)
  b, n = sorter.shape
  inv = np.empty((b, n), dtype=np.intp)
  b_indices = np.arange(b)
  inv[b_indices[:, None], sorter] = np.arange(n, dtype=np.intp) + 1
  return inv[b_indices[:, None], targets.reshape(b, -1)].reshape(targets_shape)


def load_tf_embedding(exp_id, save_dir, epoch=0, noise_multiplier=0.,
                      l2_norm_clip=0., microbatches=128):
  model_name = 'tfw2v_{}'.format(exp_id)
  vocab_path = os.path.join(save_dir, model_name, 'vocab.txt')

  if epoch > 0:
    model_name += '{}_n{}_l{}_mb{}'.format(epoch, noise_multiplier,
                                           l2_norm_clip, microbatches)
  save_path = os.path.join(save_dir, model_name)
  np_path = os.path.join(save_path, 'word_emb.npz')

  if os.path.exists(np_path):
    with np.load(np_path) as f:
      word_emb = f['arr_0']
  else:
    word_emb = load_trained_variable(os.path.abspath(save_path), 'emb')
    np.savez(np_path, word_emb)

  dictionary = dict()
  with open(vocab_path) as f:
    i = 0
    for line in f:
      word = line.split(' ')[0]
      dictionary[word] = i
      i += 1

  model_path = './models/w2v/wiki9_w2v_{}.model'.format(exp_id)
  word2vec_model = Word2Vec.load(model_path)
  vocab = word2vec_model.wv.vocab

  word2vec_model.wv.vectors = np.zeros((len(vocab), word_emb.shape[1]))
  for word in vocab:
    tf_index = dictionary[word.encode("utf-8")]
    word2vec_model.wv.vectors[vocab[word].index] = word_emb[tf_index]

  return word2vec_model


def load_glove_model(model_path):
  glove_model = Glove.load(model_path)
  word2vec_model = Word2Vec.load(model_path.replace('glove', 'w2v'))
  vocab = word2vec_model.wv.vocab

  for word in vocab:
    glove_index = glove_model.dictionary[word]
    word2vec_model.wv.vectors[vocab[word].index] = \
      glove_model.word_vectors[glove_index]

  return word2vec_model
