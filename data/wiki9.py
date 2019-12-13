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

from gensim.corpora.wikicorpus import WikiCorpus, init_to_ignore_interrupt, \
  ARTICLE_MIN_WORDS, _process_article, IGNORED_NAMESPACES, get_namespace
from pickle import PicklingError
from xml.etree.cElementTree import iterparse, ParseError

from common import gen_seed, DATA_DIR
import smart_open
import random
import gensim.utils
import os
import bz2
import multiprocessing
import logging
import tqdm

from six import raise_from

WIKI9_PATH = os.path.join(DATA_DIR, 'wiki9', 'enwik9.bz2')
WIKI9_DIR = os.path.join(DATA_DIR, 'wiki9', 'articles')
WIKI9_SPLIT_DIR = os.path.join(DATA_DIR, 'wiki9', 'split')

for d in [WIKI9_DIR, WIKI9_SPLIT_DIR]:
  if not os.path.exists(d):
    os.makedirs(d)


def extract_pages(f, filter_namespaces=False, filter_articles=None):
  try:
    elems = (elem for _, elem in iterparse(f, events=("end",)))
  except ParseError:
    yield None, "", None

  elem = next(elems)
  namespace = get_namespace(elem.tag)
  ns_mapping = {"ns": namespace}
  page_tag = "{%(ns)s}page" % ns_mapping
  text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
  title_path = "./{%(ns)s}title" % ns_mapping
  ns_path = "./{%(ns)s}ns" % ns_mapping
  pageid_path = "./{%(ns)s}id" % ns_mapping

  for elem in elems:
    if elem.tag == page_tag:
      title = elem.find(title_path).text
      text = elem.find(text_path).text

      if filter_namespaces:
        ns = elem.find(ns_path).text
        if ns not in filter_namespaces:
          text = None

      if filter_articles is not None:
        if not filter_articles(
          elem, namespace=namespace, title=title,
          text=text, page_tag=page_tag,
          text_path=text_path, title_path=title_path,
          ns_path=ns_path, pageid_path=pageid_path):
          text = None

      pageid = elem.find(pageid_path).text
      yield title, text or "", pageid  # empty page will yield None

      elem.clear()


class MyWikiCorpus(WikiCorpus):
  @staticmethod
  def save_corpus(fname, corpus, id2word=None, metadata=False):
    pass

  def get_texts(self):
    logger = logging.getLogger(__name__)

    articles, articles_all = 0, 0
    positions, positions_all = 0, 0

    tokenization_params = (
      self.tokenizer_func, self.token_min_len, self.token_max_len, self.lower)
    texts = ((text, self.lemmatize, title, pageid, tokenization_params)
             for title, text, pageid in extract_pages(bz2.BZ2File(self.fname),
                                                      self.filter_namespaces,
                                                      self.filter_articles))

    pool = multiprocessing.Pool(self.processes, init_to_ignore_interrupt)

    try:
      # process the corpus in smaller chunks of docs,
      # because multiprocessing.Pool
      # is dumb and would load the entire input into RAM at once...
      for group in gensim.utils.chunkize(texts, chunksize=10 * self.processes,
                                         maxsize=1):
        for tokens, title, pageid in pool.imap(_process_article, group):
          articles_all += 1
          positions_all += len(tokens)
          # article redirects and short stubs are pruned here
          if len(tokens) < self.article_min_tokens or \
              any(title.startswith(ignore + ':') for ignore in
                  IGNORED_NAMESPACES):
            continue
          articles += 1
          positions += len(tokens)
          yield (tokens, (pageid, title))

    except KeyboardInterrupt:
      logger.warn(
        "user terminated iteration over Wikipedia corpus after %i"
        " documents with %i positions "
        "(total %i articles, %i positions before pruning articles"
        " shorter than %i words)",
        articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
      )
    except PicklingError as exc:
      raise_from(
        PicklingError('Can not send filtering function {} to multiprocessing, '
                      'make sure the function can be pickled.'.format(
                        self.filter_articles)), exc)
    else:
      logger.info(
        "finished iterating over Wikipedia corpus of %i "
        "documents with %i positions "
        "(total %i articles, %i positions before pruning articles"
        " shorter than %i words)",
        articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
      )
      self.length = articles  # cache corpus length
    finally:
      pool.terminate()


def write_wiki9_articles():
  wiki = MyWikiCorpus(WIKI9_PATH, lemmatize=False, dictionary={},
                      filter_namespaces=False)
  for text, (p_id, title) in tqdm.tqdm(wiki.get_texts()):
    if title is None:
      continue

    article_path = os.path.join(WIKI9_DIR, p_id)
    if os.path.exists(article_path):
      continue

    with open(article_path, 'wb') as f:
      f.write(' '.join(text).encode("utf-8"))


def split_wiki9_articles(exp_id=0):
  all_docs = list(os.listdir(WIKI9_DIR))

  s = gen_seed(exp_id)
  random.seed(s)
  random.shuffle(all_docs)
  random.seed()

  n = len(all_docs) // 2
  return all_docs[:n], all_docs[n:]


def read_wiki9_train_split(exp_id=0):
  split_path = os.path.join(WIKI9_SPLIT_DIR, 'split{}.train'.format(exp_id))
  if not os.path.exists(split_path):
    train_docs, _ = split_wiki9_articles()
    with open(split_path, 'wb') as f:
      for doc in tqdm.tqdm(train_docs):
        with open(os.path.join(WIKI9_DIR, doc)) as fd:
          f.write(fd.read())
        f.write(' ')

  return split_path


class WIKI9Articles(object):
  def __init__(self, docs, dirname=WIKI9_DIR, verbose=0):
    self.docs = docs
    self.dirname = dirname
    self.verbose = verbose

  def __iter__(self):
    for fname in tqdm.tqdm(self.docs) if self.verbose else self.docs:
      for line in smart_open.open(os.path.join(self.dirname, fname),
                                  'r', encoding='utf-8'):
        yield line.split()


if __name__ == '__main__':
  split_wiki9_articles()
