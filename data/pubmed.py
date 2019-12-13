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

import re
import os
import json
import tqdm

from common import DATA_DIR
from nltk.tokenize import sent_tokenize

PUBMED_JSON_DIR = os.path.join(DATA_DIR, 'pubmed', 'json')
PUBMED_ABSTRACT_DIR = os.path.join(DATA_DIR, 'pubmed', 'abstract')


if not os.path.exists(PUBMED_ABSTRACT_DIR):
  os.makedirs(PUBMED_ABSTRACT_DIR)


def load_stop_words():
  stop_words = set()
  with open('./data/stopwords.txt') as f:
    for line in f:
      stop_words.add(line.strip())
  return stop_words


def read_pubmed_abstracts(min_len=5, max_len=50):
  json_files = os.listdir(PUBMED_JSON_DIR)
  stop_words = load_stop_words()
  for json_file in tqdm.tqdm(json_files):
    json_path = os.path.join(PUBMED_JSON_DIR, json_file)
    abstract_sents = []
    with open(json_path, 'rb') as f:
      data = json.load(f)
      for article in data:
        abstract = article['article_abstract'].lower()
        sents = sent_tokenize(abstract)
        for text in sents:
          text = re.sub(r"[0-9]+([.)])", '', text)
          text = re.sub(r"([!@*&#$^_().,;:\'\"\[\]?/\\><+=]+|[-]+ | [-]+|--)",
                        '', text)
          if text.strip():
            words = text.split()
            if len(words) < min_len or len(words) >= max_len \
                    or all(w in stop_words for w in words):
              continue

            abstract_sents.append(text)

    with open(os.path.join(PUBMED_ABSTRACT_DIR,
                           json_file.replace('json', 'txt')), 'wb') as f:
      for sent in abstract_sents:
        f.write(sent.encode('utf-8') + '\n')


if __name__ == '__main__':
  read_pubmed_abstracts()
