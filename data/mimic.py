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
import tqdm

import numpy as np
from common import DATA_DIR
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed

NOTE_DIR = os.path.join(DATA_DIR, 'patient_notes')
DIAGNOSIS_DIR = os.path.join(DATA_DIR, 'diagnosis')

if not os.path.exists(DIAGNOSIS_DIR):
  os.makedirs(DIAGNOSIS_DIR)


def remove_private(text):
  text = re.sub(r'\[\*\*[^\]]*\*\*]', '', text)
  text = text.replace('M.D.', '')
  text = text.replace('Dr.', '')
  return text.strip()


def remove_titles(text, titles):
  titles = r'|'.join(titles + [':'])
  text = re.sub(titles, '', text)
  return remove_private(text)


def process_note(note):
  title_pat = r"^.*?\s*([a-zA-Z',\.\-\*\[\]\(\) ]+):"
  diag_titles = ['discharge diagnosis', 'discharge diagnoses',
                 'final diagnosis', 'final diagnoses']
  sub_diag_titles = ['diagnosis', 'diagnoses', 'primary', 'secondary']

  def is_diag_section(t):
    return any(diag_title in t for diag_title in diag_titles)

  def is_sub_diag_section(t):
    return any(diag_title in t for diag_title in sub_diag_titles)

  with open(os.path.join(NOTE_DIR, note), 'rb') as f:
    text = f.read().lower()
    diagnosis_descriptions = []
    start_append = False
    for line in text.split('\n'):
      m = re.search(title_pat, line, re.I)

      if m and len(diagnosis_descriptions):
        temp_title = line[m.span()[0]: m.span()[1] - 1]
        if not is_sub_diag_section(temp_title):
          break
        else:
          line = remove_titles(line, sub_diag_titles)
          if line:
            diagnosis_descriptions.append(line)
      elif start_append:
        line = remove_private(line)
        if len(line) > 1:
          diagnosis_descriptions.append(line)

      if is_diag_section(line):
        start_append = True
        line = remove_titles(line, diag_titles + sub_diag_titles)
        if len(line) > 1:
          diagnosis_descriptions.append(line)

  if len(diagnosis_descriptions):
    text = ' '.join(diagnosis_descriptions)
    text = re.sub(r"[0-9]+([.)])", '', text)
    text = re.sub(r"([!@*&#$^_().,;:\'\"\[\]?/\\><+]+|[-]+ | [-]+|--)",
                  '', text)
    text = word_tokenize(text)
    text = [w for w in text if w != '-']
    if len(text):
      with open(os.path.join(DIAGNOSIS_DIR, note), 'wb') as f:
        f.write(' '.join(text))
    else:
      print(note)
      print(diagnosis_descriptions)


def read_all_patient_notes():
  all_notes = os.listdir(NOTE_DIR)
  Parallel(n_jobs=16)(delayed(process_note)(note)
                      for note in tqdm.tqdm(all_notes))


def load_diagnosis(note, max_len=30):
  with open(os.path.join(DIAGNOSIS_DIR, note)) as f:
    text = f.read()
    return text.split()[:max_len]


def load_all_diagnosis(train_size=0.5, split_word=True, seed=12345):
  all_notes = os.listdir(DIAGNOSIS_DIR)
  all_notes = sorted(all_notes)
  texts = []
  for note in tqdm.tqdm(all_notes):
    text = load_diagnosis(note)
    if not split_word:
      text = ' '.join(text)
    texts.append(text)

  n = len(texts)
  n_train = int(train_size * n)
  np.random.seed(seed)
  texts = np.asarray(texts)
  np.random.shuffle(texts)
  return texts[:n_train], texts[n_train:]


if __name__ == '__main__':
  load_all_diagnosis()
