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

from gensim.models import KeyedVectors
from data.common import WORD_EMB_PATH
from data.bookcorpus import VOCAB_PATH
import collections
import os
import tqdm
import numpy as np
import sklearn.linear_model
import tensorflow as tf


GLOVE_EMBEDDING_PATH = WORD_EMB_PATH + 'glove.840B.300d_gensim.txt'
W2V_EMBEDDING_PATH = WORD_EMB_PATH + 'GoogleNews-vectors-negative300.bin'
VOCAB_PATH = VOCAB_PATH.format(0)


def _expand_vocabulary(thought_emb, vocab, word2vec):
  # Find words shared between the two vocabularies.
  print("Finding shared words")
  shared_words = [w for w in vocab if w in word2vec]

  # Select embedding vectors for shared words.
  print("Selecting embeddings for %d shared words" % len(shared_words))
  shared_st_emb = thought_emb[[vocab[w] for w in shared_words]]
  shared_w2v_emb = word2vec[shared_words]
  print(shared_w2v_emb.shape)

  # Train a linear regression model on the shared embedding vectors.
  print("Training linear regression model")
  model = sklearn.linear_model.LinearRegression()
  model.fit(shared_w2v_emb, shared_st_emb)

  # Create the expanded vocabulary.
  print("Creating embeddings for expanded vocabuary")
  combined_emb = collections.OrderedDict()
  # Ignore words with underscores (spaces).
  all_words = [w for w in word2vec.vocab if "_" not in w]
  all_words = np.asarray(all_words)
  num_words = len(all_words)
  batch_size = 2048
  num_batches = num_words // batch_size + (num_words % batch_size != 0)

  for batch_idx in tqdm.trange(num_batches):
    ws = all_words[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    w_embs = model.predict(word2vec[ws])
    for w, w_emb in zip(ws, w_embs):
      combined_emb[w] = w_emb

  for w in vocab:
    combined_emb[w] = thought_emb[vocab[w]]

  print("Created expanded vocabulary of %d words" % len(combined_emb))
  return combined_emb


def load_bookcorpus_vocab():
  vocab = dict()
  with open(VOCAB_PATH) as f:
    i = 1
    for line in f:
      vocab[line.split("\t")[0]] = i
      i += 1
  return vocab


def load_pretrained_word_embedding(glove=False):
  print("loading pretrained embedding from disk...")
  word_emb_model = KeyedVectors.load_word2vec_format(
      GLOVE_EMBEDDING_PATH if glove else W2V_EMBEDDING_PATH,
      binary=not glove, limit=1500000)
  return word_emb_model


def expand_vocabulary(model_dir, vocab=None, emb_name="emb_in",
                      use_glove=False, save=False):

  # Load the skip-thoughts embeddings and vocabulary.
  thought_emb = tf.train.load_variable(model_dir, emb_name)

  if vocab is None:
    vocab = load_bookcorpus_vocab()

  # Load the Word2Vec model.
  word2vec = load_pretrained_word_embedding(use_glove)

  # Run vocabulary expansion.
  embedding_map = _expand_vocabulary(thought_emb, vocab, word2vec)

  # Save the output
  words = embedding_map.keys()
  embeddings = np.array(embedding_map.values())

  if save:
    output_dir = os.path.join(model_dir, "expand")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    vocab_file = os.path.join(output_dir, "vocab.txt")
    with tf.io.gfile.GFile(vocab_file, "w") as f:
      f.write("\n".join(words))
    print("Wrote vocabulary file to %s" % vocab_file)

    embeddings_file = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_file, embeddings)
    print("Wrote embeddings file to %s" % embeddings_file)

  vocab = dict()
  for i, w in enumerate(words):
    vocab[w] = i + 1

  embeddings = np.vstack([thought_emb[0], embeddings])
  return vocab, embeddings

