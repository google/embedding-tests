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

import numpy as np
import scipy.stats as stats


def interquartile_range_mean(values):
  p25 = np.percentile(values, 25)
  p75 = np.percentile(values, 75)

  iqr_values = list()
  for v in values:
    if p25 <= v <= p75:
      iqr_values.append(v)

  if len(iqr_values) == 0:
    return np.mean([p25, p75])

  return np.mean(iqr_values)


def mean_absolute_deviation(values):
  median = np.mean(values)
  dev_values = [abs(v - median) for v in values]

  return np.mean(dev_values)


def gini_coefficient(values):
  sort_values = sorted(values)
  cum_values = np.cumsum(sort_values)
  return 1.0 + 1.0 / len(values) - 2 * (sum(cum_values) /
                                        (cum_values[-1] * len(values)))


def calculate_aggregate(values):
  agg_measures = {
    'avg': np.mean(values),
    'std': np.std(values),
    'var': np.var(values),
    'med': np.median(values),
    '10p': np.percentile(values, 10),
    '25p': np.percentile(values, 25),
    '50p': np.percentile(values, 50),
    '75p': np.percentile(values, 75),
    '90p': np.percentile(values, 90),
    'iqr': np.percentile(values, 75) - np.percentile(values, 25),
    'iqm': interquartile_range_mean(values),
    'mad': mean_absolute_deviation(values),
    'gin': gini_coefficient(values),
    'skw': stats.skew(values),
    'kur': stats.kurtosis(values)
  }
  keys = sorted(agg_measures.keys())
  values = [agg_measures[key] for key in keys]
  values = np.asarray(values)
  return np.nan_to_num(values)
