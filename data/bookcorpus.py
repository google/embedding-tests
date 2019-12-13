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

from nltk.tokenize import sent_tokenize, word_tokenize
from joblib import Parallel, delayed
from collections import Counter, defaultdict
import unidecode
import numpy as np
from common import DATA_DIR, GLOVE_EMBEDDING_PATH, W2V_EMBEDDING_PATH, gen_seed

try:
  import cPickle as pkl
except ImportError:
  import pickle as pkl

import keras_preprocessing.text
import smart_open
import os
import re
import tqdm
import random
import json


BOOKCORPUS_DIR = DATA_DIR + 'bookcorpus/'
BOOKCORPUS_RAW_DIR = DATA_DIR + 'bookcorpus/out_txts/'
BOOKCORPUS_PROCESSED_DIR = DATA_DIR + 'bookcorpus/processed_txts/'
BOOKCORPUS_DATA_DIR = DATA_DIR + 'bookcorpus/data/'
BOOKCORPUS_NP_DIR = DATA_DIR + 'bookcorpus/numpy/'
BOOKCORPUS_AUTHOR_DIR = DATA_DIR + 'bookcorpus/author/'
VOCAB_PATH = os.path.join(BOOKCORPUS_DATA_DIR, 'word_count{}')

if not os.path.exists(BOOKCORPUS_PROCESSED_DIR):
  os.makedirs(BOOKCORPUS_PROCESSED_DIR)

if not os.path.exists(BOOKCORPUS_NP_DIR):
  os.makedirs(BOOKCORPUS_NP_DIR)

if not os.path.exists(BOOKCORPUS_DATA_DIR):
  os.makedirs(BOOKCORPUS_DATA_DIR)

if not os.path.exists(BOOKCORPUS_AUTHOR_DIR):
  os.makedirs(BOOKCORPUS_AUTHOR_DIR)


def validate_words(words):
  if not words:
    return False
  if len(words) == 1:
    return False
  if re.match('^[\W_]+$', words):
    return False
  for word in words.split():
    if len(word.strip()) > 20:
      return False
  return True


def tokenize_file(filename):
  raw_filename = os.path.join(BOOKCORPUS_RAW_DIR, filename)
  tokenized_sents = []
  with smart_open.open(raw_filename, encoding='utf-8') as f:
    lines = f.readlines()
    if lines[0].strip() == '<!DOCTYPE html>':
      return

    for line in lines:
      sents = sent_tokenize(line)
      for sent in sents:
        if sent.strip():
          if 'http' in sent:
            continue

          sent = sent.strip()
          sent = sent.replace(u'\u2026', '')
          sent = unidecode.unidecode_expect_nonascii(sent)
          sent = sent.replace('_', '')
          words = " ".join(word_tokenize(sent))
          words = words.replace('``', '"')
          words = words.replace('`', '\'')
          words = words.replace('\'\'', '"')

          if validate_words(words):
            tokenized_sents.append(words)

  if len(tokenized_sents) <= 1:
    return

  with smart_open.open(os.path.join(BOOKCORPUS_PROCESSED_DIR, filename),
                       'w', encoding='utf-8') as f:
    for line in tokenized_sents:
      if line[-1] != '\n':
        line += '\n'
      f.write(line.decode('utf-8'))


def process_raw_files(n_jobs=48):
  all_filenames = os.listdir(BOOKCORPUS_RAW_DIR)
  Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(tokenize_file)(filename) for filename in all_filenames)


def split_bookcorpus(exp_id=0):
  all_docs = list(os.listdir(BOOKCORPUS_PROCESSED_DIR))
  all_docs = sorted(all_docs)

  s = gen_seed(exp_id)
  random.seed(s)
  random.shuffle(all_docs)
  random.seed()

  n = len(all_docs) // 2
  return all_docs[:n], all_docs[n:]


def count_words_in_file(filename, max_len=30):
  wordcount = Counter()
  with smart_open.open(os.path.join(BOOKCORPUS_PROCESSED_DIR, filename),
                       'r', encoding='utf-8') as f:
    for line in f:
      words = line.replace('\n', '').split()
      for w in words[:max_len * 2]:
        wordcount[w] += 1
  return wordcount


def build_vocabulary(exp_id=0, n_jobs=48, vocab_size=50000,
                     rebuild=False, max_len=30):
  vocab_path = os.path.join(BOOKCORPUS_DATA_DIR, 'vocab{}.pkl'.format(exp_id))
  if not rebuild and os.path.exists(vocab_path):
    with open(vocab_path, 'rb') as f:
      vocab = pkl.load(f)
  else:
    train_docs, _ = split_bookcorpus(exp_id)
    wordcount = Counter()
    results = Parallel(n_jobs=n_jobs, verbose=1)(
      delayed(count_words_in_file)(filename, max_len)
      for filename in train_docs)

    for r in tqdm.tqdm(results):
      wordcount.update(r)

    wordcount = wordcount.most_common(vocab_size)
    vocab = dict()
    idx = 1
    for word, count in wordcount:
      vocab[word] = idx
      idx += 1

    with open(vocab_path, 'wb') as f:
      pkl.dump(vocab, f, -1)

    with open(VOCAB_PATH.format(exp_id), 'w') as f:
      for w, c, in wordcount:
        f.write('{}\t{}\n'.format(w.encode("utf-8"), c))

  return vocab


def file_to_word_ids(filename, vocab, max_len=30):
  file_inputs = []
  file_masks = []
  with smart_open.open(os.path.join(BOOKCORPUS_PROCESSED_DIR, filename),
                       'r', encoding='utf-8') as f:
    for line in f:
      words = line.replace('\n', '').split()

      word_ids = [vocab.get(word, 0) for word in words]
      masks = [1] * len(word_ids)

      if sum(word_ids) == 0:  # all unknowns
        continue

      if len(word_ids) < max_len:
        pads = [0] * (max_len - len(word_ids))
        word_ids.extend(pads)
        masks.extend(pads)

      file_inputs.append(word_ids[:max_len])
      file_masks.append(masks[:max_len])

  if len(file_inputs) == 0:
    print(filename)
    return

  file_inputs = np.asarray(file_inputs, dtype=np.int32)
  file_masks = np.asarray(file_masks, dtype=np.int8)

  save_path = os.path.join(BOOKCORPUS_NP_DIR, filename + '.npz')
  np.savez(save_path, file_inputs, file_masks)


def preprocess_bookcorpus_sentences(exp_id=0, n_jobs=48,
                                    vocab_size=50000, max_len=30):
  train_docs, test_docs = split_bookcorpus(exp_id)
  vocab = build_vocabulary(exp_id, n_jobs, vocab_size, rebuild=True,
                           max_len=max_len)
  Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(file_to_word_ids)(filename, vocab, max_len=max_len)
    for filename in train_docs)
  Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(file_to_word_ids)(filename, vocab, max_len=max_len)
    for filename in test_docs)


def load_numpy_file(filename):
  with np.load(os.path.join(BOOKCORPUS_NP_DIR, filename + '.npz')) as f:
    file_inputs, file_masks = f['arr_0'], f['arr_1']
  return file_inputs, file_masks


def load_pretrained_word_embedding(glove=False):
  from gensim.models import KeyedVectors
  print("loading pretrained embedding from disk...")
  word_emb_model = KeyedVectors.load_word2vec_format(
    GLOVE_EMBEDDING_PATH if glove else W2V_EMBEDDING_PATH, binary=not glove)
  return word_emb_model


def save_existed_word_embedding(exp_id=0, glove=True):
  vocab = build_vocabulary(exp_id, rebuild=False)

  def get_rand_vec():
    return np.random.uniform(-0.1, 0.1, size=(300, )).astype(np.float32)

  word_vectors = np.zeros((len(vocab) + 1, 300), dtype=np.float32)
  word_vectors[0] = get_rand_vec()

  word_emb_model = load_pretrained_word_embedding(glove)
  index2word = dict((idx, word) for word, idx in vocab.items())

  missed = 0
  for i in range(1, len(index2word) + 1):
    word = index2word[i]
    if word not in word_emb_model:
      word_vectors[i] = get_rand_vec()
      missed += 1
    else:
      word_vectors[i] = word_emb_model.wv[word].astype(np.float32)

  print('Missed {} of {} words'.format(missed, len(vocab)))
  np.savez(os.path.join(BOOKCORPUS_DATA_DIR, '{}{}.npz'.format(
    'glove' if glove else 'w2v', exp_id)), word_vectors)


def load_initialized_word_emb(exp_id=0, glove_only=False):
  save_path = os.path.join(BOOKCORPUS_DATA_DIR, 'glove{}.npz'.format(exp_id))
  with np.load(save_path) as f:
    glove = f['arr_0']

  if glove_only:
    return glove

  save_path = os.path.join(BOOKCORPUS_DATA_DIR, 'w2v{}.npz'.format(exp_id))
  with np.load(save_path) as f:
    w2v = f['arr_0']
  return np.hstack([glove, w2v])


def load_bookcorpus_sentences(exp_id=0, test_mi=False, load_author=False):
  train_docs, test_docs = split_bookcorpus(exp_id)

  if load_author:
    book_authors = load_book_author(train_docs)
    unique_authors = np.unique(book_authors.values())
    authors_to_ids = dict(zip(unique_authors, np.arange(len(unique_authors))))
    book_author_ids = dict((k, authors_to_ids[v])
                           for k, v in book_authors.items())

  def load_data(filenames, rtn_author=False):
    inputs, masks = [], []
    if rtn_author:
      authors = []

    for filename in tqdm.tqdm(filenames):
      file_inputs, file_masks = load_numpy_file(filename)
      inputs.append(file_inputs)
      masks.append(file_masks)
      if rtn_author:
        authors.append([book_author_ids[filename]] * len(file_inputs))

    if not test_mi:
      inputs, masks = np.vstack(inputs), np.vstack(masks)
      if rtn_author:
        return inputs, masks, np.concatenate(authors)
      else:
        return inputs, masks
    else:
      return inputs, masks

  train_data = load_data(train_docs, rtn_author=load_author)
  train_inputs, train_masks = train_data[:2]
  vocab = build_vocabulary(exp_id, rebuild=False)

  if not test_mi:
    if load_author:
      return train_inputs, train_masks, train_data[2], vocab
    else:
      return train_inputs, train_masks, vocab

  test_inputs, test_masks = load_data(test_docs)
  return train_inputs, train_masks, test_inputs, test_masks, vocab


def remove_duplicate_author(author):
  if 'Harun Yahya' in author or 'Adnan Oktar' in author:
    return 'Harun Yahya'
  return author


def load_book_author(filenames=None):
  if filenames is None:
    filenames = set(os.listdir(BOOKCORPUS_PROCESSED_DIR))

  meta_path = os.path.join(BOOKCORPUS_DIR, 'url_list.jsonl')
  book_attributes = dict()

  with smart_open.open(meta_path, encoding='utf-8') as f:
    for line in f:
      data = json.loads(line.strip())
      _, book_id = os.path.split(data['page'])
      _, file_name = os.path.split(data['epub'])
      book_name = '{}__{}'.format(book_id, file_name.replace('.epub', '.txt'))
      if book_name in filenames:
        book_attributes[book_name] = remove_duplicate_author(data['author'])

  return book_attributes


def load_book_categories(filenames=None):
  if filenames is None:
    filenames = set(os.listdir(BOOKCORPUS_PROCESSED_DIR))

  meta_path = os.path.join(BOOKCORPUS_DIR, 'url_list.jsonl')
  book_attributes = dict()

  with smart_open.open(meta_path, encoding='utf-8') as f:
    for line in f:
      data = json.loads(line.strip())
      _, book_id = os.path.split(data['page'])
      _, file_name = os.path.split(data['epub'])
      book_name = '{}__{}'.format(book_id, file_name.replace('.epub', '.txt'))
      if book_name in filenames:
        if len(data['genres']) == 0:
          category = 'Other'
        else:
          category = data['genres'][0].replace('\n', '').strip()
        book_attributes[book_name] = category

  return book_attributes


def load_data_with_attribute(attribute, test_only=True, train_only=False):
  train_filenames, test_filenames = split_bookcorpus(0)
  if test_only:
    all_filenames = test_filenames
  elif train_only:
    all_filenames = train_filenames
  else:
    all_filenames = train_filenames + test_filenames

  if attribute == 'category':
    attribute_dict = load_book_categories()
  elif attribute == 'book_id':
    unique_book_ids = np.sort(all_filenames)
    attribute_dict = dict(zip(unique_book_ids, unique_book_ids))
  elif attribute == 'author':
    attribute_dict = load_book_author()
  else:
    raise ValueError('Wrong attribute')

  return all_filenames, attribute_dict


def preprocess_pipeline(n_jobs=48, start_from_raw=False):
  if start_from_raw:
    process_raw_files(n_jobs=n_jobs)
  preprocess_bookcorpus_sentences(n_jobs=n_jobs)
  save_existed_word_embedding(glove=True)
  save_existed_word_embedding(glove=False)


def filter_tokenized_file(filename, vocab, min_len=5):
  sents = []
  filters = {'copyright', 'chapter', 'edition', 'license', 'licensed',
             'published'}
  with smart_open.open(os.path.join(BOOKCORPUS_PROCESSED_DIR, filename),
                       encoding='utf-8') as f:
    for line in f:
      sent = line.replace('\n', '')
      sent = sent.replace(u'\u2026', '')
      sent = unidecode.unidecode_expect_nonascii(sent)
      sent = sent.replace('``', '"')
      sent = sent.replace('`', '\'')
      sent = sent.replace('\'\'', '"')

      words = sent.split()

      if len(words) < min_len:
        continue

      num_punk = 0
      num_known = 0
      filter_flag = False

      for word in words:
        if word.lower() in filters:
          filter_flag = True
          break

        if not word.isalnum():
          num_punk += 1

        elif word in vocab:
          num_known += 1

      if filter_flag:
        continue

      thresh = len(words) * 0.5
      if num_punk >= thresh or num_known < 1:
        continue

      sents.append(sent)

  if len(sents) > 0:
    with smart_open.open(os.path.join(BOOKCORPUS_AUTHOR_DIR, filename), 'w',
                         encoding='utf-8') as f:
      for sent in sents:
        f.write(sent.decode('utf-8') + '\n')


def authorship_filter(exp_id=0):
  _, filenames = split_bookcorpus(exp_id)
  vocab = build_vocabulary(exp_id, rebuild=False)
  Parallel(n_jobs=32)(
      delayed(filter_tokenized_file)(filename, vocab)
      for filename in tqdm.tqdm(filenames))


def load_single_author_file(filename, split_word, remove_punct, min_len=0):
  sents = []
  with smart_open.open(os.path.join(BOOKCORPUS_AUTHOR_DIR, filename),
                       encoding='utf-8') as f:
    for line in f:
      sent = line.replace('\n', '')
      if remove_punct:
        sent = keras_preprocessing.text.text_to_word_sequence(sent, lower=False)
      else:
        sent = sent.split()

      if len(sent) < min_len:
        continue

      if not split_word:
        sent = " ".join(sent)

      sents.append(sent)
    return sents


def load_author_data(min_book=2, train_size=50, test_size=300,
                     unlabeled_size=0, top_attr=0, split_by_book=True,
                     split_word=True, remove_punct=False, seed=54321,
                     min_len=0):

  min_size = max(test_size + unlabeled_size, test_size * 2)
  if split_by_book:
    # use sentence from one book to infer authorship of other books
    assert min_book >= 2

  filenames = os.listdir(BOOKCORPUS_AUTHOR_DIR)
  attribute_dict = load_book_author(filenames)
  author_book_count = Counter(attribute_dict.values())
  filtered_filenames = [fname for fname in filenames if
                        author_book_count[attribute_dict[fname]] >= min_book]

  if remove_punct:
    file_sents = Parallel(n_jobs=8)(
      delayed(load_single_author_file)(fname, split_word, remove_punct, min_len)
      for fname in tqdm.tqdm(filtered_filenames)
    )
  else:
    file_sents = [load_single_author_file(fname, split_word,
                                          remove_punct, min_len)
                  for fname in tqdm.tqdm(filtered_filenames)]

  author_file_sents = defaultdict(list)
  author_sent_counts = Counter()

  for fname, sents in zip(filtered_filenames, file_sents):
    author = attribute_dict[fname]
    author_file_sents[author].append(sents)
    author_sent_counts[author] += len(sents)

  authors = [tup[0] for tup in author_sent_counts.most_common()]
  filtered_authors = []
  for author in authors:
    author_books = author_file_sents[author]
    num_books = len(author_books)
    assert num_books >= min_book
    if all(len(sents) >= min_size for sents in author_books):
      filtered_authors.append(author)
      if 0 < top_attr <= len(filtered_authors):
        break

  train_sents, train_authors = [], []
  test_sents, test_authors = [], []
  unlabeled_sents, unlabeled_authors = [], []

  np.random.seed(seed)
  for author in filtered_authors:
    author_books = author_file_sents[author]
    num_books = len(author_books)

    if split_by_book:
      train_author_book_idx = np.random.choice(num_books)
      train_author_book = author_books[train_author_book_idx]
      train_author_sents = np.random.choice(train_author_book,
                                            size=train_size, replace=False)
      rest_author_book_indices = [i for i in range(num_books)
                                  if i != train_author_book_idx]
      rest_author_books = np.concatenate([author_books[i] for i in
                                          rest_author_book_indices])
      rest_author_sents = np.random.choice(rest_author_books,
                                           size=min_size, replace=False)
    else:
      all_author_sents = np.concatenate(author_books)
      sampled_author_sents = np.random.choice(all_author_sents,
                                              size=train_size + min_size,
                                              replace=False)
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


if __name__ == '__main__':
  authorship_filter()
  # load_author_data()
