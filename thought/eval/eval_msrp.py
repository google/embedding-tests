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

# Evaluation for MSRP

import numpy as np
from data_handler import load_msrp as load_data
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1


def evaluate(encoder, k=10, seed=1234, evalcv=False, evaltest=True,
             use_feats=True):
  """
  Run experiment
  k: number of CV folds
  test: whether to evaluate on test set
  """
  traintext, testtext, labels = load_data()

  trainA = encoder.encode(traintext[0], verbose=False, norm=True)
  trainB = encoder.encode(traintext[1], verbose=False, norm=True)

  if evalcv:
    print 'Running cross-validation...'
    C = eval_kfold(trainA, trainB, traintext, labels[0], shuffle=True, k=k,
                   seed=seed, use_feats=use_feats)
  else:
    C = 4

  if evaltest:
    print 'Computing testing skipthoughts...'
    testA = encoder.encode(testtext[0], verbose=False, norm=True)
    testB = encoder.encode(testtext[1], verbose=False, norm=True)

    if use_feats:
      train_features = np.c_[
        np.abs(trainA - trainB), trainA * trainB, feats(traintext[0],
                                                        traintext[1])]
      test_features = np.c_[
        np.abs(testA - testB), testA * testB, feats(testtext[0], testtext[1])]
    else:
      train_features = np.c_[np.abs(trainA - trainB), trainA * trainB]
      test_features = np.c_[np.abs(testA - testB), testA * testB]

    print 'Evaluating...'
    clf = LogisticRegression(C=C)
    clf.fit(train_features, labels[0])
    yhat = clf.predict(test_features)
    acc = clf.score(test_features, labels[1])
    f1_score = f1(labels[1], yhat)
    print 'Test accuracy: ' + str(acc)
    print 'Test F1: ' + str(f1_score)
    return acc, f1_score


def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def feats(A, B):
  """
  Compute additional features (similar to Socher et al.)
  These alone should give the same result from their paper (~73.2 Acc)
  """
  tA = [t.split() for t in A]
  tB = [t.split() for t in B]

  nA = [[w for w in t if is_number(w)] for t in tA]
  nB = [[w for w in t if is_number(w)] for t in tB]

  features = np.zeros((len(A), 6))

  # n1
  for i in range(len(A)):
    if set(nA[i]) == set(nB[i]):
      features[i, 0] = 1.

  # n2
  for i in range(len(A)):
    if set(nA[i]) == set(nB[i]) and len(nA[i]) > 0:
      features[i, 1] = 1.

  # n3
  for i in range(len(A)):
    if set(nA[i]) <= set(nB[i]) or set(nB[i]) <= set(nA[i]):
      features[i, 2] = 1.

  # n4
  for i in range(len(A)):
    features[i, 3] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tA[i]))

  # n5
  for i in range(len(A)):
    features[i, 4] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tB[i]))

  # n6
  for i in range(len(A)):
    features[i, 5] = 0.5 * (
          (1.0 * len(tA[i]) / len(tB[i])) + (1.0 * len(tB[i]) / len(tA[i])))

  return features


def eval_kfold(A, B, train, labels, shuffle=True, k=10, seed=1234,
               use_feats=False):
  """
  Perform k-fold cross validation
  """
  # features
  labels = np.array(labels)
  if use_feats:
    features = np.c_[np.abs(A - B), A * B, feats(train[0], train[1])]
  else:
    features = np.c_[np.abs(A - B), A * B]

  scan = [2 ** t for t in range(0, 9, 1)]
  kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
  scores = []

  for s in scan:
    scanscores = []
    for train, test in kf.split(features):
      # Split data
      X_train = features[train]
      y_train = labels[train]
      X_test = features[test]
      y_test = labels[test]

      # Train classifier
      clf = LogisticRegression(C=s)
      clf.fit(X_train, y_train)
      yhat = clf.predict(X_test)
      fscore = f1(y_test, yhat)
      scanscores.append(fscore)

    # Append mean score
    scores.append(np.mean(scanscores))
    print scores

  # Get the index of the best score
  s_ind = np.argmax(scores)
  s = scan[int(s_ind)]
  print(s)
  return s
