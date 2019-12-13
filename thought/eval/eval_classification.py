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

# Experiment scripts for binary classification benchmarks
# (e.g. MR, CR, MPQA, SUBJ)

import numpy as np
import data_handler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def eval_nested_kfold(encoder, name, k=10, seed=1234, verbose=False, s=16):
  """
  Evaluate features with nested K-fold cross validation
  Outer loop: Held-out evaluation
  Inner loop: Hyperparameter tuning

  Datasets can be found at http://nlp.stanford.edu/~sidaw/home/projects:nbsvm
  Options for name are 'MR', 'CR', 'SUBJ' and 'MPQA'
  """
  # Load the dataset and extract features
  z, features = data_handler.load_data(encoder, name, seed=seed)

  scan = [2 ** t for t in range(2, 8, 1)]
  kf = KFold(k, random_state=seed)
  scores = []

  for train, test in kf.split(features):
    # Split data
    X_train = features[train]
    y_train = z['labels'][train]
    X_test = features[test]
    y_test = z['labels'][test]

    # def inner_kfold(C):
    #   # Inner KFold
    #   innerkf = KFold(k, random_state=seed + 1)
    #   innerscores = []
    #   for innertrain, innertest in innerkf.split(X_train):
    #
    #     # Split data
    #     X_innertrain = X_train[innertrain]
    #     y_innertrain = y_train[innertrain]
    #     X_innertest = X_train[innertest]
    #     y_innertest = y_train[innertest]
    #
    #     # Train classifier
    #     clf = LogisticRegression(C=C)
    #     clf.fit(X_innertrain, y_innertrain)
    #     acc = clf.score(X_innertest, y_innertest)
    #     innerscores.append(acc)
    #
    #   return np.mean(innerscores)
    #
    # scanscores = Parallel(len(scan))(delayed(inner_kfold)(s) for s in scan)
    # # Get the index of the best score
    # s_ind = np.argmax(scanscores)
    # s = scan[int(s_ind)]

    # Train classifier
    clf = LogisticRegression(C=s)
    clf.fit(X_train, y_train)

    # Evaluate
    acc = clf.score(X_test, y_test)
    scores.append(acc)
    if verbose:
      print(s, acc)

  score = np.mean(scores)
  print(score)
  return score
