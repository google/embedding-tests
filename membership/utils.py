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

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from aggregate_stats import calculate_aggregate
from joblib import Parallel, delayed

import tqdm
import numpy as np


def advantage_from_preds(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  return tpr - fpr


def advantage_from_scores(y_true, y_score):
  fprs, tprs, threshs = roc_curve(y_true, y_score)
  advs = tprs - fprs
  best_idx = np.argmax(advs)
  best_adv, best_thresh = advs[best_idx], threshs[best_idx]
  return best_adv, best_thresh


def compute_adversarial_advantage(train_metrics, test_metrics):
  print('Computing adversarial advantage for {} train and {} unseen'.format(
    len(train_metrics), len(test_metrics)))
  avg_thresh = np.mean(train_metrics)
  y_score = np.concatenate([train_metrics, test_metrics])
  y_pred = y_score >= avg_thresh
  y_true = np.concatenate([np.ones_like(train_metrics),
                           np.zeros_like(test_metrics)])

  print('Average membership AUC: {:.4f}'.format(roc_auc_score(y_true, y_score)))
  avg_adv = advantage_from_preds(y_true, y_pred)
  print('Average adversarial advantage: {:.4f} with threshold {:.2f}'.format(
    avg_adv, avg_thresh))

  best_adv, best_thresh = advantage_from_scores(y_true, y_score)
  print('Best adversarial advantage: {:.4f} with threshold {:.2f}'.format(
    best_adv, best_thresh))
  return best_adv


def histogram_feats(metrics, range, bins=100, density=False):
  feats, _ = np.histogram(metrics, bins=bins, density=density, range=range)
  return feats


def adversarial_advantage_from_trained(data, metric_range=None, train_size=0.1,
                                       verbose=False, histogram=True, n_jobs=8,
                                       model='svc', norm=False, scale=False):
  def metric_to_feat(m):
    if histogram:
      f = histogram_feats(m, metric_range, bins=100, density=False)
    else:
      f = calculate_aggregate(m)
    return f

  def collect_feats(ms):
    if n_jobs == 1:
      fs = []
      for m in tqdm.tqdm(ms) if verbose else ms:
        fs.append(metric_to_feat(m))
      return np.asarray(fs, dtype=np.float)
    else:
      fs = Parallel(n_jobs, verbose=verbose)(delayed(metric_to_feat)(m)
                                             for m in ms)
      return np.asarray(fs, dtype=np.float)

  if len(data) == 2:
    train_metrics, test_metrics = data
    train_feats = collect_feats(train_metrics)
    test_feats = collect_feats(test_metrics)
    n = min(len(train_feats), len(test_feats))
    x = np.vstack([train_feats[:n], test_feats[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    x_train, x_test, y_train, y_test = train_test_split(
      x, y, train_size=train_size, stratify=y)
  else:
    assert len(data) == 4
    train_m_metrics, train_nm_metrics, test_m_metrics, test_nm_metrics = data
    train_m_feats = collect_feats(train_m_metrics)
    test_m_feats = collect_feats(test_m_metrics)
    train_nm_feats = collect_feats(train_nm_metrics)
    test_nm_feats = collect_feats(test_nm_metrics)
    x_train = np.vstack([train_m_feats, train_nm_feats])
    y_train = np.concatenate([np.ones(len(train_m_feats)),
                              np.zeros(len(train_nm_feats))])
    x_test = np.vstack([test_m_feats, test_nm_feats])
    y_test = np.concatenate([np.ones(len(test_m_feats)),
                             np.zeros(len(test_nm_feats))])

  y_pred, y_score = train_clf(x_train, y_train, x_test, y_test, verbose, model,
                              norm=norm, scale=scale)

  print('trained membership AUC: {:.4f}'.format(roc_auc_score(y_test, y_score)))

  adv = advantage_from_preds(y_test, y_pred)
  print('Trained adversarial advantage: {:.4f}'.format(adv))

  best_adv, best_thresh = advantage_from_scores(y_test, y_score)
  print('Best trained adversarial advantage: {:.4f} with threshold'
        ' {:.2f}'.format(best_adv, best_thresh))
  return adv


def train_clf(x_train, y_train, x_test, y_test, verbose=False, model='svc',
              norm=False, scale=False):
  if norm:
    norm = Normalizer()
    x_train = norm.fit_transform(x_train)
    x_test = norm.transform(x_test)

  if scale:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

  if model == 'svc':
    clf = LinearSVC(verbose=verbose, max_iter=3000)
  elif model == 'log':
    clf = LogisticRegression(verbose=verbose, max_iter=3000)
  elif model == 'rf':
    clf = RandomForestClassifier(verbose=verbose, n_estimators=100,
                                 n_jobs=16)
  else:
    raise ValueError(model)

  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)

  if verbose:
    print(classification_report(y_test, y_pred))

  if isinstance(clf, LinearSVC):
    y_score = clf.decision_function(x_test)
  else:
    y_score = clf.predict_proba(x_test)[:, 1]
  return y_pred, y_score
