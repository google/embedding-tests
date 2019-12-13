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
import os
import random

DATA_DIR = '/path/to/data/'
MODEL_DIR = '/path/to/model/'
EMB_DIR = '/path/to/save/embedding/'

WORD_EMB_PATH = '/path/to/word/embedding/'
GLOVE_EMBEDDING_PATH = WORD_EMB_PATH + 'glove.840B.300d_gensim.txt'
W2V_EMBEDDING_PATH = WORD_EMB_PATH + 'GoogleNews-vectors-negative300.bin'


def gen_seed(idx, n=100):
  random.seed(12345)

  seeds = []
  for i in range(n):
    s = random.random()
    seeds.append(s)

  return seeds[idx]


def load_pretrained_word_embedding(glove=False):
  from gensim.models import KeyedVectors
  print("loading pretrained embedding from disk...")
  word_emb_model = KeyedVectors.load_word2vec_format(
    GLOVE_EMBEDDING_PATH if glove else W2V_EMBEDDING_PATH, binary=not glove)
  return word_emb_model


def get_pretrained_word_embedding(vocab, glove=True,
                                  vocab_include_pad=False, save_path=None):
  if save_path is not None and os.path.exists(save_path):
    print('Load word embedding from', save_path)
    with np.load(save_path) as f:
      return f['arr_0']

  vocab_size = len(vocab)
  if not vocab_include_pad:
    vocab_size += 1

  def get_rand_vec():
    return np.random.uniform(-0.1, 0.1, size=(300, )).astype(np.float32)

  word_vectors = np.zeros((vocab_size, 300), dtype=np.float32)
  word_vectors[0] = get_rand_vec()

  word_emb_model = load_pretrained_word_embedding(glove)
  index2word = dict((idx, word) for word, idx in vocab.items())

  missed = 0
  for i in range(1, vocab_size):
    word = index2word[i]
    if word not in word_emb_model:
      word_vectors[i] = get_rand_vec()
      missed += 1
    else:
      word_vectors[i] = word_emb_model.wv[word].astype(np.float32)

  print('Missed {} of {} words'.format(missed, len(vocab)))
  if save_path is not None:
    np.savez(save_path, word_vectors)

  return word_vectors
