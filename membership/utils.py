from __future__ import division
from __future__ import print_function

import numpy as np

import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import LinearSVC

from aggregate_stats import calculate_aggregate


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
                                       verbose=False, histogram=True, n_jobs=4):
  def metric_to_feat(m):
    if histogram:
      f = histogram_feats(m, metric_range, bins=100, density=True)
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
    train_feats, test_feats = collect_feats(train_metrics),\
                              collect_feats(test_metrics)
    n = min(len(train_feats), len(test_feats))
    x = np.vstack([train_feats[:n], test_feats[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    x_train, x_test, y_train, y_test = train_test_split(
      x, y, train_size=train_size, stratify=y)
  else:
    assert len(data) == 4
    train_member_metrics, train_nonmember_metrics,\
    test_member_metrics, test_nonmember_metrics = data
    train_m_feats, test_m_feats = collect_feats(train_member_metrics), \
                                  collect_feats(test_member_metrics)
    train_nm_feats, test_nm_feats = collect_feats(train_nonmember_metrics), \
                                    collect_feats(test_nonmember_metrics)
    x_train = np.vstack([train_m_feats, train_nm_feats])
    y_train = np.concatenate([np.ones(len(train_m_feats)),
                              np.zeros(len(train_nm_feats))])
    x_test = np.vstack([test_m_feats, test_nm_feats])
    y_test = np.concatenate([np.ones(len(test_m_feats)),
                             np.zeros(len(test_nm_feats))])
  if not histogram:
    norm = Normalizer()
    x_train = norm.fit_transform(x_train)
    x_test = norm.transform(x_test)

  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)

  clf = LinearSVC(verbose=verbose, max_iter=3000)
  clf.fit(x_train, y_train)

  y_pred = clf.predict(x_test)
  if verbose:
    print(classification_report(y_test, y_pred))

  adv = advantage_from_preds(y_test, y_pred)
  print('Trained adversarial advantage: {:.4f}'.format(adv))

  y_score = clf.decision_function(x_test)
  best_adv, best_thresh = advantage_from_scores(y_test, y_score)
  print('Best trained adversarial advantage: {:.4f} with threshold'
        ' {:.2f}'.format(best_adv, best_thresh))
  return adv
