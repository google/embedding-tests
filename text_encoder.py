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

import tensorflow as tf
import tqdm
import os

from thought.image_text_model import ImageTextEmbedding
from thought.quick_thought_model import QuickThoughtModel
from utils.common_utils import log
from utils.sent_utils import iterate_minibatches_indices
from collections import defaultdict


class QuickThoughtEncoder(object):
  def __init__(self, vocab_size, emb_dim=620, encoder_dim=1200, context_size=1,
               cell_type='LSTM', num_layer=3, init_word_emb=None):
    self.model = QuickThoughtModel(vocab_size, emb_dim, encoder_dim,
                                   context_size, num_layer=num_layer,
                                   init_word_emb=init_word_emb,
                                   cell_type=cell_type, train=False)

    self.inputs_a = tf.placeholder(tf.int64, (None, None), name="inputs_a")
    self.masks_a = tf.placeholder(tf.int32, (None, None), name="masks_a")

    self.inputs_b = tf.placeholder(tf.int64, (None, None), name="inputs_b")
    self.masks_b = tf.placeholder(tf.int32, (None, None), name="masks_b")

    self.encoded_a = self.encode_fn(self.inputs_a, self.masks_a)
    self.encoded_b = self.encode_fn(self.inputs_b, self.masks_b)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

  def load_weights(self, model_path):
    self.saver.restore(self.sess, model_path)

  def encode_fn(self, x, m):
    encode_emb = tf.nn.embedding_lookup(self.model.word_in_emb, x)
    encoded = self.model.encode(encode_emb, m, self.model.in_cells,
                                self.model.proj_in)
    return encoded

  def encode(self, inputs_a, masks_a, query_size):
    feats_a = []
    for idx in iterate_minibatches_indices(len(inputs_a), query_size):
      feed = {self.inputs_a: inputs_a[idx], self.masks_a: masks_a[idx]}
      feat_a = self.sess.run(self.encoded_a, feed_dict=feed)
      feats_a.append(feat_a)

    feats_a = np.vstack(feats_a)
    return feats_a

  def encode_tuple(self, inputs_a, masks_a, inputs_b, masks_b, query_size):
    feats_a, feats_b = [], []
    for idx in iterate_minibatches_indices(len(inputs_a), query_size):
      feed = {self.inputs_a: inputs_a[idx], self.masks_a: masks_a[idx],
              self.inputs_b: inputs_b[idx], self.masks_b: masks_b[idx]}

      feat_a, feat_b = self.sess.run([self.encoded_a, self.encoded_b],
                                     feed_dict=feed)
      feats_a.append(feat_a)
      feats_b.append(feat_b)

    feats_a = np.vstack(feats_a)
    feats_b = np.vstack(feats_b)
    return feats_a, feats_b


class ImageTextEncoder(object):
  def __init__(self, word_emb, encoder_dim, encoder_type='rnn', norm=True,
               **kwargs):
    self.layer_idx = kwargs.get('layer_idx', -1)
    self.model = ImageTextEmbedding(word_emb, encoder_dim,
                                    encoder_type=encoder_type,
                                    norm=norm, drop_p=0.0)

    self.feats = tf.placeholder(tf.float32, shape=(None, 2048), name='feats')
    self.caps = tf.placeholder(tf.int64, shape=(None, None), name='caps')
    self.masks = tf.placeholder(tf.bool, shape=(None, None), name='masks')
    training = tf.constant(False, tf.bool)
    img_emb, text_emb = self.model.encode(self.feats, self.caps,
                                          self.masks, training)
    all_text_layers = self.model.text_outputs + [text_emb]

    self.fetches = [img_emb, all_text_layers[self.layer_idx]]
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

  def load_weights(self, model_path):
    print('Loading weight from {}'.format(model_path))
    self.saver.restore(self.sess, os.path.join(model_path, 'model.ckpt'))

  def encode(self, data):
    names = np.sort(data[0].keys())
    n_data = len(names)
    img_features, img_caption_ids, captions, caption_masks = data

    # each image id is associated with a list of text ids
    img_id_text_ids = defaultdict(list)
    img_offset = 0
    text_offset = 0

    # collect embeddings
    img_embs, text_embs = [], []
    pbar = tqdm.tqdm(total=n_data)
    for batch_idx in iterate_minibatches_indices(n_data, 1024, False, True):
      batch_feats = []
      batch_caps = []
      batch_masks = []

      for i, img_name in enumerate(names[batch_idx]):
        batch_feats.append(img_features[img_name])
        text_idx = img_caption_ids[img_name]
        batch_caps.append(captions[text_idx])
        batch_masks.append(caption_masks[text_idx])

        # update the id dict
        n_texts_per_image = len(text_idx)
        img_id_text_ids[img_offset] = [text_offset + j
                                       for j in range(n_texts_per_image)]
        img_offset += 1
        text_offset += n_texts_per_image

      feed_dict = {
        self.feats: np.vstack(batch_feats),
        self.caps: np.vstack(batch_caps),
        self.masks: np.vstack(batch_masks)
      }
      batch_img_embs, batch_text_embs = self.sess.run(self.fetches, feed_dict)

      img_embs.append(batch_img_embs)
      text_embs.append(batch_text_embs)
      pbar.update(len(batch_img_embs))

    pbar.close()
    img_embs = np.vstack(img_embs)
    text_embs = np.vstack(text_embs)

    return img_embs, text_embs, img_id_text_ids


LocalEncoders = {
    'quickthought':
        lambda v: QuickThoughtEncoder(vocab_size=v),
    'transformer':
        lambda v: QuickThoughtEncoder(
            vocab_size=v,
            cell_type='TRANS',
            encoder_dim=600,
            emb_dim=600,
            num_layer=3),
}


def preprocess_raw_data(sents, vocab, max_len):
  x = np.zeros((len(sents), max_len), dtype=np.int64)
  m = np.zeros((len(sents), max_len), dtype=np.int32)

  for i, sent in enumerate(sents):
    # sent = sent.lower()
    if isinstance(vocab, dict):
      word_indices = [vocab.get(w, 0) for w in sent]
    else:
      word_indices = vocab.encode(sent.decode('utf-8'))

    if sum(word_indices) == 0:
      raise ValueError('Should have at least one known word {}'.format(sent))

    length = min(len(word_indices), max_len)
    x[i, :length] = word_indices[:length]
    m[i, :length] = 1
  return x, m


def encode_sentences(vocab, model_path, config, *sents, **kwargs):
  max_len = kwargs.get('max_len', 30)
  high_layer_idx = kwargs.get('high_layer_idx', -1)
  low_layer_idx = kwargs.get('low_layer_idx', -1)
  query_size = kwargs.get('query_size', 2048)

  log('Encoding sentences on the fly...')
  vocab_size = len(vocab) + 1 if isinstance(vocab, dict) else vocab.vocab_size

  model = QuickThoughtModel(vocab_size, config['emb_dim'],
                            config['encoder_dim'], 1,  init_word_emb=None,
                            cell_type=config['cell_type'],
                            num_layer=config['num_layer'], train=False)

  inputs = tf.placeholder(tf.int64, (None, None), name='inputs')
  masks = tf.placeholder(tf.int32, (None, None), name='masks')

  encode_emb = tf.nn.embedding_lookup(model.word_in_emb, inputs)
  all_layers = model.encode(encode_emb, masks, model.in_cells, model.proj_in,
                            return_all_layers=True)

  learn_mapping = high_layer_idx != low_layer_idx
  if high_layer_idx == low_layer_idx:
    encoded = all_layers[high_layer_idx]
  else:
    encoded = (all_layers[low_layer_idx], all_layers[high_layer_idx])

  model_vars = tf.trainable_variables()
  saver = tf.train.Saver(model_vars)
  sess = tf.Session()

  saver.restore(sess, model_path)
  encoder_fn = lambda s: sess.run(encoded, {inputs: s[0], masks: s[1]})

  def encode_sents(s, n):
    embs_low, embs_high = [], []
    pbar = tqdm.tqdm(total=n)
    for batch_idx in iterate_minibatches_indices(n, query_size, False):
      emb = encoder_fn((s[0][batch_idx], s[1][batch_idx]))
      if learn_mapping:
        embs_low.append(emb[0])
        embs_high.append(emb[1])
        n_batch = len(emb[0])
      else:
        embs_low.append(emb)
        n_batch = len(emb)
      pbar.update(n_batch)

    pbar.close()
    if learn_mapping:
      return np.vstack(embs_low), np.vstack(embs_high)
    else:
      return np.vstack(embs_low)

  rtn_data = []
  for sent in sents:
    n_sent = len(sent)
    y, m = preprocess_raw_data(sent, vocab, max_len)
    x = encode_sents((y, m), n_sent)
    data = (x, y, m)
    rtn_data.append(data)

  return rtn_data


def encode_parsed_sentences(config, model_path, *data, **kwargs):
  high_layer_idx = kwargs.get('high_layer_idx', -1)
  low_layer_idx = kwargs.get('low_layer_idx', -1)
  query_size = kwargs.get('query_size', 2048)

  log('Encoding sentences on the fly...')
  model = QuickThoughtModel(config['vocab_size'], config['emb_dim'],
                            config['encoder_dim'], 1,  init_word_emb=None,
                            cell_type=config['cell_type'],
                            num_layer=config['num_layer'], train=False)

  inputs = tf.placeholder(tf.int64, (None, None), name='inputs')
  masks = tf.placeholder(tf.int32, (None, None), name='masks')

  encode_emb = tf.nn.embedding_lookup(model.word_in_emb, inputs)
  all_layers = model.encode(encode_emb, masks, model.in_cells, model.proj_in,
                            return_all_layers=True)

  learn_mapping = high_layer_idx != low_layer_idx
  if high_layer_idx == low_layer_idx:
    encoded = all_layers[high_layer_idx]
  else:
    encoded = (all_layers[low_layer_idx], all_layers[high_layer_idx])

  model_vars = tf.trainable_variables()
  saver = tf.train.Saver(model_vars)
  sess = tf.Session()

  saver.restore(sess, model_path)
  encoder_fn = lambda s: sess.run(encoded, {inputs: s[0], masks: s[1]})

  def encode_sents(s, n):
    embs_low, embs_high = [], []
    pbar = tqdm.tqdm(total=n)
    for batch_idx in iterate_minibatches_indices(n, query_size, False):
      emb = encoder_fn((s[0][batch_idx], s[1][batch_idx]))
      if learn_mapping:
        embs_low.append(emb[0])
        embs_high.append(emb[1])
        n_batch = len(emb[0])
      else:
        embs_low.append(emb)
        n_batch = len(emb)
      pbar.update(n_batch)

    pbar.close()
    if learn_mapping:
      return np.vstack(embs_low), np.vstack(embs_high)
    else:
      return np.vstack(embs_low)

  rtn_data = []
  for y, m in data:
    n_sent = len(y)
    x = encode_sents((y, m.astype(np.int32)), n_sent)
    rtn_data.append(x)

  return rtn_data

