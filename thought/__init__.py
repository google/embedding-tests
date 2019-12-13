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


def get_quickthought_model_name(**kwargs):
  if 'batch_size' not in kwargs:
    kwargs['batch_size'] = 800

  if 'gamma' in kwargs and kwargs['gamma'] > 0.:
    return get_quickthought_adv_model_name(**kwargs)

  model_name = 'bookcorpus_e{}_LSTM_b{}'.format(kwargs['epoch'],
                                                kwargs['batch_size'])
  if 'context' in kwargs and kwargs['context']:
    model_name += '_context'

  return model_name


def get_quickthought_adv_model_name(**kwargs):
  assert kwargs['gamma'] > 0.
  model_name = 'bookcorpus_e{}_LSTM_b{}_{}_adv{}'.format(
      kwargs['epoch'], kwargs['batch_size'], kwargs['attr'], kwargs['gamma'])
  return model_name


def get_transformer_model_name(**kwargs):
  if 'num_layer' not in kwargs:
    kwargs['num_layer'] = 3

  if 'batch_size' not in kwargs:
    kwargs['batch_size'] = 800

  if 'gamma' in kwargs and kwargs['gamma'] > 0.:
    return get_transformer_adv_model_name(**kwargs)

  model_name = 'bookcorpus_e{}_TRANSl{}_b{}'.format(
    kwargs['epoch'], kwargs['num_layer'], kwargs['batch_size'])
  return model_name


def get_transformer_adv_model_name(**kwargs):
  assert kwargs['gamma'] > 0.
  model_name = 'bookcorpus_e{}_TRANSl{}_b{}_{}_adv{}'.format(
      kwargs['epoch'],  kwargs['num_layer'], kwargs['batch_size'],
      kwargs['attr'], kwargs['gamma'])
  return model_name


def get_conversation_model_name(**kwargs):
  if 'num_layer' not in kwargs:
    kwargs['num_layer'] = 3

  kwargs['batch_size'] = 800
  model_name = 'reddit_e{}_TRANSl{}_b{}_spmFalse'.format(
    kwargs['epoch'], kwargs['num_layer'], kwargs['batch_size'])
  return model_name


ThoughtModelNameFunc = {
  'quickthought': get_quickthought_model_name,
  'transformer': get_transformer_model_name,
  'conversation': get_conversation_model_name,
}


def get_model_config(model_name):
  config = {}
  if model_name == 'quickthought':
    emb_dim, encoder_dim, cell_type, vocab_size = 620, 1200, 'LSTM', 50001
  elif model_name == 'transformer':
    emb_dim, encoder_dim, cell_type, vocab_size = 600, 600, 'TRANS', 50001
  elif model_name == 'conversation':
    emb_dim, encoder_dim, cell_type, vocab_size = 512, 512, 'TRANS', 32681
  else:
    raise ValueError(model_name)

  config['emb_dim'] = emb_dim
  config['encoder_dim'] = encoder_dim
  config['cell_type'] = cell_type
  config['num_layer'] = 3
  config['vocab_size'] = vocab_size
  return config


def get_model_ckpt_name(model_name, **kwargs):
  assert model_name in ThoughtModelNameFunc
  if 'epoch' not in kwargs:
    kwargs['epoch'] = 0

  return ThoughtModelNameFunc[model_name](**kwargs)
