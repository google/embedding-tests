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


def get_quickthought_model_name(epoch=0, model_type='LSTM', batch_size=800,
                                gamma=0.):
  if gamma == 0.:
    model_name = 'bookcorpus_e{}_{}_b{}'.format(epoch, model_type, batch_size)
  else:
    model_name = 'bookcorpus_e{}_{}_b{}_adv{}'.format(epoch, model_type,
                                                      batch_size, gamma)
  return model_name


def get_skipthought_model_name(epoch=0, model_type='GRU', batch_size=128):
  model_name = 'bookcorpus_e{}_{}_b{}_skip'.format(epoch, model_type,
                                                   batch_size)
  return model_name


def get_transformer_model_name(epoch=0, num_layer=3, batch_size=800):
  model_name = 'bookcorpus_e{}_TRANSl{}_b{}'.format(
    epoch, num_layer, batch_size)
  return model_name


ThoughtModelNameFunc = {
  'quickthought': get_quickthought_model_name,
  'skipthought': get_skipthought_model_name,
  'transformer': get_transformer_model_name,
}
