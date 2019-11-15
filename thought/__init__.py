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
