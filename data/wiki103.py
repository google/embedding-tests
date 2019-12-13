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
import smart_open
from nltk.tokenize import sent_tokenize

from bookcorpus import build_vocabulary, DATA_DIR

WIKI103_PATH = os.path.join(DATA_DIR, 'wiki103', 'wiki.train.tokens')
WIKI103_CROSS_PATH = os.path.join(DATA_DIR, 'wiki103', 'wiki.cross.tokens')


def save_cross_domain_data(min_len=5):
  vocab = build_vocabulary(exp_id=0, rebuild=False)
  cross_domain_sents = []
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
            _ = [vocab[w] for w in words]
          except KeyError:
            continue

          cross_domain_sents.append(sent)

  with smart_open.open(WIKI103_CROSS_PATH, 'w', encoding='utf-8') as f:
    for line in cross_domain_sents:
      f.write(line + '\n')


def read_raw_cross_domain_data(size):
  sents = []
  with smart_open.open(WIKI103_CROSS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
      sents.append(line.strip())
      if len(sents) >= size:
        break
  return sents


def load_wiki_cross_domain_data(size, split_word):
  sents = []
  with smart_open.open(WIKI103_CROSS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
      sent = line.strip()
      if split_word:
        sent = sent.split()
      sents.append(sent)
      if len(sents) >= size:
        break
  return sents


if __name__ == '__main__':
  save_cross_domain_data()
