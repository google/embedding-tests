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

import numpy as np
import os

import smart_open
import tqdm
from nltk.tokenize import sent_tokenize

from bookcorpus import build_vocabulary
from common import DATA_DIR

WIKI103_PATH = os.path.join(DATA_DIR, 'wiki103', 'wiki.train.tokens')
NUM_SHARD = 20


def load_cross_domain_data(min_len=5, max_len=30):
  vocab = build_vocabulary(exp_id=0, rebuild=False)
  n = 0
  cross_domain_sents = []
  cross_domain_masks = []
  with smart_open.open(WIKI103_PATH, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.replace('\n', '')
      if line.strip() and '=' not in line:
        sents = sent_tokenize(line)
        for sent in sents:
          words = sent.split()

          if len(words) < min_len:
            continue

          try:
            idx = [vocab[w] for w in words]
          except KeyError:
            continue

          n += 1
          s = np.zeros(max_len, dtype=np.int64)
          m = np.zeros(max_len, dtype=np.int32)

          if len(idx) > max_len:
            idx = idx[:max_len]

          s[:len(idx)] = idx
          m[:len(idx)] = 1

          cross_domain_sents.append(s)
          cross_domain_masks.append(m)

  cross_domain_sents = np.vstack(cross_domain_sents)
  cross_domain_masks = np.vstack(cross_domain_masks)

  save_path = os.path.join(DATA_DIR, 'wiki103', 'wiki_invert.npz')
  np.savez(save_path, cross_domain_sents, cross_domain_masks)


def load_cross_domain_context_data(min_len=5, max_len=30):
  vocab = build_vocabulary(exp_id=0, rebuild=False)
  n = 0
  cross_domain_sents_a = []
  cross_domain_masks_a = []
  cross_domain_sents_b = []
  cross_domain_masks_b = []

  def unknown_check(idx):
    return sum(np.equal(idx, 0)) > 1

  def idx_to_array(idx):
    s = np.zeros(max_len, dtype=np.int64)
    m = np.zeros(max_len, dtype=np.int32)
    s[:len(idx)] = idx
    m[:len(idx)] = 1
    return s, m

  with smart_open.open(WIKI103_PATH, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.replace('\n', '')
      if line.strip() and '=' not in line:
        sents = sent_tokenize(line)
        if len(sents) <= 2:
          continue

        for sent_a, sent_b in zip(sents[:-1], sents[1:]):
          words_a = sent_a.split()
          words_b = sent_b.split()

          if len(words_a) < min_len or len(words_b) < min_len:
            continue

          idx_a = [vocab.get(w, 0) for w in words_a][:max_len]
          idx_b = [vocab.get(w, 0) for w in words_b][:max_len]

          if unknown_check(idx_a) or unknown_check(idx_b):
            continue

          s_a, m_a = idx_to_array(idx_a)
          s_b, m_b = idx_to_array(idx_b)

          if np.array_equal(s_a, s_b):
            continue

          cross_domain_sents_a.append(s_a)
          cross_domain_masks_a.append(m_a)
          cross_domain_sents_b.append(s_b)
          cross_domain_masks_b.append(m_b)
          n += 1

  cross_domain_sents_a = np.vstack(cross_domain_sents_a)
  cross_domain_masks_a = np.vstack(cross_domain_masks_a)
  cross_domain_sents_b = np.vstack(cross_domain_sents_b)
  cross_domain_masks_b = np.vstack(cross_domain_masks_b)

  save_path = os.path.join(DATA_DIR, 'wiki103', 'wiki_invert_ctx.npz')
  np.savez(save_path, cross_domain_sents_a, cross_domain_masks_a,
           cross_domain_sents_b, cross_domain_masks_b)


def load_cross_domain_invert_data(model_name):
  n_shards = NUM_SHARD
  save_dir = os.path.join(DATA_DIR, 'wiki103', model_name)

  save_path = os.path.join(DATA_DIR, 'wiki103', 'wiki_invert.npz')
  with np.load(save_path) as f:
    cross_domain_sents = f['arr_0']
    cross_domain_masks = f['arr_1']

  cross_domain_embs = []
  for i in tqdm.trange(n_shards):
    save_path = os.path.join(save_dir, 'wiki_emb{}-{}.npz'.format(i + 1,
                             n_shards))
    with np.load(save_path) as f:
      embs = f['arr_0']

    cross_domain_embs.append(embs)

  cross_domain_embs = np.vstack(cross_domain_embs)
  return cross_domain_embs, cross_domain_sents, cross_domain_masks


def load_cross_domain_invert_context_data(model_name, pre_as_target=True):
  if model_name == 'handout':
    n_shards = 100
    save_dir = os.path.join(DATA_DIR, 'wiki103', 'wiki103_handout')
  else:
    n_shards = NUM_SHARD
    save_dir = os.path.join(DATA_DIR, 'wiki103')

  save_path = os.path.join(DATA_DIR, 'wiki103', 'wiki_invert_ctx.npz')
  with np.load(save_path) as f:
    cross_domain_sents_a, cross_domain_masks_a = f['arr_0'], f['arr_1']
    cross_domain_sents_b, cross_domain_masks_b = f['arr_2'], f['arr_3']

  def load_embs(name='a'):
    cross_domain_embs = []
    for i in tqdm.trange(n_shards):
      path = os.path.join(save_dir, 'wiki_ctx_{}_{}_emb{}-{}.npz'.format(
        name, model_name, i + 1, n_shards))
      with np.load(path) as f:
        embs = f['arr_0']
      cross_domain_embs.append(embs)
    return np.vstack(cross_domain_embs)

  cross_domain_embs_a = load_embs('a')
  cross_domain_embs_b = load_embs('b')

  if pre_as_target:  # predict previous words
    return cross_domain_embs_b, cross_domain_sents_a, cross_domain_masks_a
  else:   # predict after words
    return cross_domain_embs_a, cross_domain_sents_b, cross_domain_masks_b


if __name__ == '__main__':
  load_cross_domain_data()