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

from absl import flags
from absl import app

import logging
import os
from gensim.models import Word2Vec, FastText
from glove import Corpus, Glove
from data.common import MODEL_DIR
from data.wiki9 import split_wiki9_articles, WIKI9Articles

flags.DEFINE_integer('exp_id', 0, 'Experiment trial number')
flags.DEFINE_string('model', 'w2v', 'Word embedding model')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 'w2v'),
                    'Model directory for embedding model')
FLAGS = flags.FLAGS


def load_glove_dictionary(exp_id, save_dir):
  model_path = os.path.join(save_dir, 'wiki9_w2v_{}.model'.format(exp_id))
  vocab = Word2Vec.load(model_path).wv.vocab
  dictionary = dict()
  for word in vocab:
    dictionary[word] = vocab[word].index
  return dictionary


def train_glove(corpus, params, exp_id, save_dir, save_dict=False):
  dictionary = load_glove_dictionary(exp_id, save_dir)
  # Build the corpus dictionary and the cooccurrence matrix.
  print('Pre-processing corpus')

  dict_path = os.path.join(save_dir, 'glove_dict_{}.model'.format(exp_id))
  if os.path.exists(dict_path):
    corpus_model = Corpus.load(dict_path)
  else:
    corpus_model = Corpus(dictionary)
    corpus_model.fit(corpus, window=params['window'] * 2, ignore_missing=True)
    if save_dict:
      corpus_model.save(dict_path)

  print('Dict size: %s' % len(corpus_model.dictionary))
  print('Collocations: %s' % corpus_model.matrix.nnz)

  glove = Glove(no_components=100, learning_rate=params['alpha'])
  glove.fit(corpus_model.matrix, epochs=50, no_threads=params['workers'],
            verbose=True)
  glove.add_dictionary(corpus_model.dictionary)
  return glove


def train_word_embedding(exp_id=0, emb_model='w2v'):
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                      level=logging.INFO)

  params = {
    'sg': 1,
    'negative': 25,
    'alpha': 0.05,
    'sample': 1e-4,
    'workers': 48,
    'iter': 5,
    'window': 5,
  }

  train_docs, test_docs = split_wiki9_articles(exp_id)
  print(len(train_docs), len(test_docs))
  wiki9_articles = WIKI9Articles(train_docs)
  save_dir = FLAGS.save_dir
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  if emb_model == 'w2v':
    model = Word2Vec(wiki9_articles, **params)
  elif emb_model == 'ft':
    model = FastText(wiki9_articles, **params)
  elif emb_model == 'glove':
    model = train_glove(wiki9_articles, params, exp_id,
                        save_dir, save_dict=True)
  else:
    raise ValueError('No such embedding model: {}'.format(emb_model))

  model_path = os.path.join(save_dir,
                            'wiki9_{}_{}.model'.format(emb_model, exp_id))
  model.save(model_path)


def main(unused_argv):
  train_word_embedding(FLAGS.exp_id, FLAGS.model)


if __name__ == '__main__':
  app.run(main)