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

# Dataset handler for binary classification tasks (MR, CR, SUBJ, MQPA)

import numpy as np
from numpy.random import RandomState
import os.path
from nltk.tokenize import word_tokenize

DATA_DIR = '/mnt/nfs/downstream/'


def load_data(encoder, name, loc=DATA_DIR, seed=1234):
    """
    Load one of MR, CR, SUBJ or MPQA
    """
    z = {}
    if name == 'MR':
        pos, neg = load_rt(loc=loc)
    elif name == 'SUBJ':
        pos, neg = load_subj(loc=loc)
    elif name == 'CR':
        pos, neg = load_cr(loc=loc)
    elif name == 'MPQA':
        pos, neg = load_mpqa(loc=loc)
    else:
        raise ValueError(name)

    labels = compute_labels(pos, neg)
    text, labels = shuffle_data(pos+neg, labels, seed=seed)
    z['text'] = text
    z['labels'] = labels
    print 'Computing skip-thought vectors...'
    features = encoder.encode(text, verbose=False)
    return z, features


def load_rt(loc='./data/'):
    """
    Load the MR dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'MR', 'rt-polarity.pos'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'MR', 'rt-polarity.neg'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_subj(loc='./data/'):
    """
    Load the SUBJ dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'SUBJ', 'subj.objective'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'SUBJ', 'subj.subjective'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_cr(loc='./data/'):
    """
    Load the CR dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'CR', 'custrev.pos'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'CR', 'custrev.neg'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def load_mpqa(loc='./data/'):
    """
    Load the MPQA dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'MPQA', 'mpqa.pos'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'MPQA', 'mpqa.neg'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def load_msrp(loc=DATA_DIR):
  """
  Load MSRP dataset
  """
  trainloc = os.path.join(loc, 'MRPC', 'msr_paraphrase_train.txt')
  testloc = os.path.join(loc, 'MRPC', 'msr_paraphrase_test.txt')

  trainA, trainB, testA, testB = [], [], [], []
  trainS, devS, testS = [], [], []

  f = open(trainloc, 'rb')
  for line in f:
    text = line.strip().split('\t')
    trainA.append(' '.join(word_tokenize(text[3])))
    trainB.append(' '.join(word_tokenize(text[4])))
    trainS.append(text[0])
  f.close()
  f = open(testloc, 'rb')
  for line in f:
    text = line.strip().split('\t')
    testA.append(' '.join(word_tokenize(text[3])))
    testB.append(' '.join(word_tokenize(text[4])))
    testS.append(text[0])
  f.close()

  trainS = [int(s) for s in trainS[1:]]
  testS = [int(s) for s in testS[1:]]

  return [trainA[1:], trainB[1:]], [testA[1:], testB[1:]], [trainS, testS]


def load_trec(loc=DATA_DIR):
  """
  Load the TREC question-type dataset
  """
  train, test = [], []
  with open(os.path.join(loc, 'TREC', 'train_5500.label'), 'rb') as f:
    for line in f:
      train.append(line.strip())
  with open(os.path.join(loc, 'TREC', 'TREC_10.label'), 'rb') as f:
    for line in f:
      test.append(line.strip())
  return train, test


def compute_labels(pos, neg):
    """
    Construct list of labels
    """
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1.0
    labels[len(pos):] = 0.0
    return labels


def shuffle_data(X, L, seed=1234):
    """
    Shuffle the data
    """
    prng = RandomState(seed)
    inds = np.arange(len(X))
    prng.shuffle(inds)
    X = [X[i] for i in inds]
    L = L[inds]
    return X, L
