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
import smart_open
from joblib import Parallel, delayed
from bookcorpus import build_vocabulary
from common import DATA_DIR
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter


REDDIT_DIR = DATA_DIR + 'reddit/'
RAW_DATA_DIR = DATA_DIR + 'reddit/shard_by_author/'
PROCESSED_DATA_DIR = DATA_DIR + 'reddit/shard_by_author_processed/'


if not os.path.exists(PROCESSED_DATA_DIR):
  os.makedirs(PROCESSED_DATA_DIR)


def translate(t):
  t = t.replace(u'\u2018', '\'')
  t = t.replace(u'\u2019', '\'')
  t = t.replace(u'\u201c', '\"')
  t = t.replace(u'\u201d', '\"')
  t = t.replace(u'\u2013', '-')
  t = t.replace(u'\u2014', '-')

  t = t.replace(u'\u2026', '')
  t = t.replace(u'\ufffd', '')
  t = t.replace(u'\ufe0f', '')
  t = t.replace(u'\u035c', '')
  t = t.replace(u'\u0296', '')
  t = t.replace(u'\u270a', '')
  t = t.replace(u'*', '')
  t = t.replace(u'~', '')

  t = t.replace(u'\ufb00', 'ff')

  return t


def preprocess(t):
  words = t.split(' ')
  for i in range(len(words)):
    if 'http' in words[i]:
      words[i] = ''
  return ' '.join(words)


def remove_puncs(words):
  new_words = []
  for w in words:
    flag = False
    for c in w:
      if c.isalnum():
        flag = True
        break
    if flag:
      new_words.append(w)
  return new_words


def write_processed_comments(user, vocab, min_len=15, max_len=64,
                             min_lines=300):
  new_lines = []
  current_lines = set()
  with smart_open.open(os.path.join(RAW_DATA_DIR, user), encoding='utf-8') as f:
    for line in f:
      text = line[1:-2].decode('unicode_escape')
      text = translate(text)
      text = preprocess(text)
      for sent in sent_tokenize(text):
        words = word_tokenize(sent)
        # words = remove_puncs(words)
        if min_len <= len(words) <= max_len:
          if any([w not in vocab for w in words]):
            continue
          new_line = ' '.join(words)
          if new_line not in current_lines:
            current_lines.add(new_line)
            new_lines.append(new_line)

  if len(new_lines) >= min_lines:
    with smart_open.open(os.path.join(PROCESSED_DATA_DIR, user),
                         'w', encoding='utf-8') as f:
      for line in new_lines:
        f.write(line + '\n')


def preprocess_comments():
  vocab = build_vocabulary(rebuild=False)
  # for user in os.listdir(RAW_DATA_DIR):
  #   write_processed_comments(user, vocab)
  Parallel(n_jobs=32, verbose=True)(
    delayed(write_processed_comments)(user, vocab)
    for user in os.listdir(RAW_DATA_DIR))


def count_author_lines():
  author_lines = Counter()
  for user in os.listdir(PROCESSED_DATA_DIR):
    filename = os.path.join(PROCESSED_DATA_DIR, user)
    with open(filename) as f:
      author_lines[user] = len(f.readlines())

  with open(os.path.join(REDDIT_DIR, 'author_lines.txt'), 'w') as f:
    for author, n_lines in author_lines.most_common():
      f.write('{}\t{}\n'.format(author, n_lines))


def load_author_lines():
  author_lines = Counter()
  with open(os.path.join(REDDIT_DIR, 'author_lines.txt')) as f:
    for line in f:
      arr = line.split('\t')
      cnt = arr[-1]
      author = ''.join(arr[:-1])
      cnt = int(cnt)
      author_lines[author] = cnt

  return author_lines


def load_single_author_file(user, split_word):
  sents = []
  with smart_open.open(os.path.join(PROCESSED_DATA_DIR, user),
                       encoding='utf-8') as f:
    for line in f:
      sent = line.strip()
      if split_word:
        sent = sent.split()
      # sent = keras_preprocessing.text.text_to_word_sequence(sent, lower=False)
      # if not split_word:
      #   sent = " ".join(sent)

      sents.append(sent)
    return sents


def load_author_data(train_size=50, test_size=100, unlabeled_size=0,
                     top_attr=100, split_word=True, seed=12345):

  min_size = max(test_size + unlabeled_size, test_size * 2)

  author_sent_counts = load_author_lines()
  filtered_authors = []
  for author, count in author_sent_counts.most_common():
    if count >= min_size:
      filtered_authors.append(author)
      if 0 < top_attr <= len(filtered_authors):
        break

  train_sents, train_authors = [], []
  test_sents, test_authors = [], []
  unlabeled_sents, unlabeled_authors = [], []

  np.random.seed(seed)
  for author in filtered_authors:
    author_sents = load_single_author_file(author, split_word)
    all_author_sents = np.asarray(author_sents)
    # sampled_author_sents = np.random.choice(all_author_sents,
    #                                         size=train_size + min_size,
    #                                         replace=False)
    sampled_author_sents = all_author_sents[:train_size + min_size]
    rest_author_sents = sampled_author_sents[:min_size]
    train_author_sents = sampled_author_sents[min_size: min_size + train_size]

    test_author_sents = rest_author_sents[:test_size]
    if unlabeled_size > 0:
      unlabeled_author_sents = rest_author_sents[test_size:
                                                 test_size + unlabeled_size]
      unlabeled_sents.append(unlabeled_author_sents)
      unlabeled_authors.extend([author] * len(unlabeled_author_sents))

    train_sents.append(train_author_sents)
    train_authors.extend([author] * len(train_author_sents))
    test_sents.append(test_author_sents)
    test_authors.extend([author] * len(test_author_sents))

  train_sents = np.concatenate(train_sents)
  test_sents = np.concatenate(test_sents)

  if len(unlabeled_sents) > 0:
    unlabeled_sents = np.concatenate(unlabeled_sents)

  return train_sents, train_authors, test_sents, test_authors, \
         unlabeled_sents, unlabeled_authors


def load_reddit_cross_domain_data(size, split_word):
  author_sent_counts = load_author_lines()
  sents = []
  for author, cnt in author_sent_counts.most_common():
    author_sents = load_single_author_file(author, split_word)
    sents += author_sents
    if len(sents) >= size:
      break
  return sents

