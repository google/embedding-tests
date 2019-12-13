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

from transformer_layers import TransformerBlock
import tensorflow as tf


def mean_pool(x, m):
  m = tf.cast(m, tf.float32)
  x = tf.multiply(x, tf.expand_dims(m, 2))
  x = tf.reduce_sum(x, 1) / tf.reduce_sum(m, 1, keepdims=True)
  return x


class RNN(object):
  def __init__(self, num_units):
    self.rnn_fw = tf.keras.layers.CuDNNLSTM(units=num_units // 2,
                                            return_sequences=True,
                                            go_backwards=False,
                                            name='rnn_fw')
    self.rnn_bw = tf.keras.layers.CuDNNLSTM(units=num_units // 2,
                                            return_sequences=True,
                                            go_backwards=False,
                                            name='rnn_bw')

  def forward(self, inputs, masks):
    def rnn_fn(x, m, rnn):
      x = rnn(x)
      # x = tf.reduce_max(x, 1)   # max pooling
      # x = mean_pool(x, m) # mean pooling
      indices = tf.reduce_sum(m, 1, keepdims=True) - 1
      x = tf.gather_nd(x, tf.cast(indices, tf.int32), batch_dims=1)
      return x

    lengths = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
    masks = tf.cast(masks, tf.float32)
    inputs = tf.multiply(inputs, tf.expand_dims(masks, 2))
    inputs_bw = tf.reverse_sequence(inputs, lengths, 1, 0)
    outputs_fw = rnn_fn(inputs, masks, self.rnn_fw)
    outputs_bw = rnn_fn(inputs_bw, masks, self.rnn_bw)
    outputs = tf.concat([outputs_fw, outputs_bw], axis=1)
    return outputs


class Transformer(object):
  def __init__(self, num_units):
    self.hidden = tf.keras.layers.Dense(num_units)
    self.transformer = TransformerBlock(num_units, num_units * 4,
                                        num_layer=2)

  def forward(self, inputs, masks):
    masks = tf.cast(masks, tf.float32)
    inputs = tf.multiply(inputs, tf.expand_dims(masks, 2))
    inputs = self.hidden(inputs)
    return self.transformer.forward(inputs, masks)


class DAN(object):
  def __init__(self, num_units):
    self.hidden = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)

  def forward(self, inputs, masks):
    masks = tf.cast(masks, tf.float32)
    inputs = tf.multiply(inputs, tf.expand_dims(masks, 2))
    inputs = tf.reduce_sum(inputs, 1) / tf.reduce_sum(masks, 1, keepdims=True)
    return self.hidden(inputs)


def get_text_encoder(encoder_type='rnn'):
  if encoder_type == 'rnn':
    return RNN
  elif encoder_type == 'trans':
    return Transformer
  elif encoder_type == 'dan':
    return DAN
  else:
    raise ValueError(encoder_type)


class ImageTextEmbedding(object):
  def __init__(self, word_emb, encoder_dim, encoder_type='rnn', norm=True,
               drop_p=0.25, contrastive=False, margin=0.5, num_neg_sample=10,
               lambda1=1.0, lambda2=1.0, internal=True):
    self.word_emb = tf.Variable(tf.convert_to_tensor(word_emb), name="emb",
                                trainable=True)
    self.text_encoder = get_text_encoder(encoder_type)(encoder_dim)

    self.text_feat_proj = tf.keras.layers.Dense(encoder_dim)
    self.img_feat_proj = tf.keras.layers.Dense(encoder_dim)
    self.dropout = tf.keras.layers.Dropout(drop_p)
    self.margin = margin
    self.num_neg_sample = num_neg_sample
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.contrastive = contrastive
    self.internal = internal
    self.norm = norm  # normalize the embedding
    self.text_outputs = []

  def forward_img(self, img_inputs, training):
    x = self.img_feat_proj(img_inputs)
    if self.norm:
      x = tf.nn.l2_normalize(x, axis=-1)
    return self.dropout(x, training=training)

  def forward_text(self, text_inputs, text_masks, training):
    if len(text_inputs.get_shape()) == 2:
      x = tf.nn.embedding_lookup(self.word_emb, text_inputs)
    else:
      x = text_inputs

    self.text_outputs.append(mean_pool(x, text_masks))
    x = self.text_encoder.forward(x, text_masks)
    self.text_outputs.append(x)
    x = self.text_feat_proj(x)
    if self.norm:
      x = tf.nn.l2_normalize(x, axis=-1)
    return self.dropout(x, training=training)

  def encode(self, img_inputs, text_inputs, text_masks, training):
    img_feats = self.forward_img(img_inputs, training)
    text_feats = self.forward_text(text_inputs, text_masks, training)
    return img_feats, text_feats

  def forward(self, img_inputs, text_inputs, text_masks, labels, training):
    img_feats, text_feats = self.encode(img_inputs, text_inputs,
                                        text_masks, training)
    if self.contrastive:
      loss = contrastive_loss(img_feats, text_feats, self.margin)
      sent_im_dist = - similarity_fn(text_feats, img_feats)
    elif self.internal:
      loss = internal_loss(img_feats, text_feats, labels)
      sent_im_dist = - similarity_fn(text_feats, img_feats)
    else:
      loss = embedding_loss(img_feats, text_feats, labels, self.margin,
                            self.num_neg_sample, self.lambda1, self.lambda2)
      sent_im_dist = pdist(text_feats, img_feats)
    rec = recall_k(sent_im_dist, labels, ks=[1, 5, 10])
    return loss, rec


def order_sim(im, s):
  im = tf.expand_dims(im, 0)
  s = tf.expand_dims(s, 1)
  diff = tf.clip_by_value(s - im, 0, 1e6)
  dist = tf.sqrt(tf.reduce_sum(diff ** 2, 2))
  scores = -tf.transpose(dist)
  return scores


def similarity_fn(im, s, order=False):
  if order:
    return order_sim(im, s)
  return tf.matmul(im, s, transpose_b=True)


def internal_loss(im_embeds, sent_embeds, im_labels):
  logits_s = tf.matmul(sent_embeds, im_embeds, transpose_b=True)
  cost_s = tf.nn.softmax_cross_entropy_with_logits_v2(im_labels, logits_s)
  logits_im = tf.matmul(im_embeds, sent_embeds, transpose_b=True)
  cost_im = tf.nn.softmax_cross_entropy_with_logits_v2(tf.transpose(im_labels),
                                                       logits_im)
  return tf.reduce_mean(cost_s) + tf.reduce_mean(cost_im)


def contrastive_loss(im_embeds, sent_embeds, margin, max_violation=True):
  """ modified https://github.com/fartashf/vsepp/blob/master/model.py#L260 """
  scores = similarity_fn(im_embeds, sent_embeds)
  batch_size = tf.shape(im_embeds)[0]
  diagonal = tf.diag_part(scores)
  d1 = tf.reshape(diagonal, (batch_size, 1))
  d2 = tf.reshape(diagonal, (1, batch_size))
  cost_s = tf.clip_by_value(margin + scores - d1, 0, 1e6)
  cost_im = tf.clip_by_value(margin + scores - d2, 0, 1e6)
  zeros = tf.zeros(batch_size)
  cost_s = tf.matrix_set_diag(cost_s, zeros)
  cost_im = tf.matrix_set_diag(cost_im, zeros)
  if max_violation:
    cost_s = tf.reduce_max(cost_s, 1)
    cost_im = tf.reduce_max(cost_im, 0)
  return tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)


def pdist(x1, x2):
  """
      x1: Tensor of shape (h1, w)
      x2: Tensor of shape (h2, w)
      Return pairwise distance for each row vector in x1, x2 as
      a Tensor of shape (h1, h2)
  """
  x1_square = tf.reshape(tf.reduce_sum(x1 * x1, axis=1), [-1, 1])
  x2_square = tf.reshape(tf.reduce_sum(x2 * x2, axis=1), [1, -1])
  return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square +
                 1e-4)


def embedding_loss(im_embeds, sent_embeds, im_labels, margin, num_neg_sample,
                   lambda1, lambda2):
  """
      im_embeds: (b, 512) image embedding tensors
      sent_embeds: (sample_size * b, 512) sentence embedding tensors
          where the order of sentence corresponds to the order of images and
          setnteces for the same image are next to each other
      im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
          True if and only if sentence[i], image[j] is a positive pair
  """
  im_labels = tf.cast(im_labels, tf.bool)
  # compute embedding loss
  num_img = tf.shape(im_embeds)[0]
  num_sent = tf.shape(sent_embeds)[0]
  sent_im_ratio = tf.div(num_sent, num_img)

  sent_im_dist = pdist(sent_embeds, im_embeds)
  # image loss: sentence, positive image, and negative image
  pos_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, im_labels),
                             [num_sent, 1])
  neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~im_labels),
                             [num_sent, -1])
  im_loss = tf.clip_by_value(margin + pos_pair_dist - neg_pair_dist,
                             0, 1e6)
  im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=num_neg_sample)[0])

  # sentence loss: image, positive sentence, and negative sentence
  neg_pair_dist = tf.reshape(
      tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.transpose(im_labels)),
      [num_img, -1])
  neg_pair_dist = tf.reshape(
      tf.tile(neg_pair_dist, [1, sent_im_ratio]), [num_sent, -1])
  sent_loss = tf.clip_by_value(margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
  sent_loss = tf.reduce_mean(tf.nn.top_k(sent_loss, k=num_neg_sample)[0])

  # sentence only loss (neighborhood-preserving constraints)
  sent_sent_dist = pdist(sent_embeds, sent_embeds)
  sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels),
                                      [1, sent_im_ratio]),
                              [num_sent, num_sent])
  pos_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, sent_sent_mask),
                             [-1, sent_im_ratio])
  pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
  neg_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, ~sent_sent_mask),
                             [num_sent, -1])
  sent_only_loss = tf.clip_by_value(margin + pos_pair_dist - neg_pair_dist,
                                    0, 1e6)
  sent_only_loss = tf.reduce_mean(tf.nn.top_k(sent_only_loss,
                                              k=num_neg_sample)[0])

  loss = im_loss * lambda1 + sent_loss + sent_only_loss * lambda2
  return loss


def recall_k(sent_im_dist, im_labels, ks=(1, 5, 10)):
  """
      Compute recall at given ks.
  """
  im_labels = tf.cast(im_labels, tf.bool)

  def retrieval_recall(dist, labels, k):
    # Use negative distance to find the index of
    # the smallest k elements in each row.
    pred = tf.nn.top_k(-dist, k=k)[1]
    # Create a boolean mask for each column (k value) in pred,
    # s.t. mask[i][j] is 1 iff pred[i][k] = j.
    pred_k_mask = lambda topk_idx: tf.one_hot(topk_idx, tf.shape(labels)[1],
                                              on_value=True, off_value=False,
                                              dtype=tf.bool)

    # Create a boolean mask for the predicted indices
    # by taking logical or of boolean masks for each column,
    # s.t. mask[i][j] is 1 iff j is in pred[i].
    pred_mask = tf.reduce_any(tf.map_fn(
      pred_k_mask, tf.transpose(pred), dtype=tf.bool), axis=0)

    # pred_mask = tf.map_fn(create_pred_mask, pred)
    # Entry (i, j) is matched iff pred_mask[i][j] and labels[i][j] are 1.
    matched = tf.cast(tf.logical_and(pred_mask, labels), dtype=tf.float32)
    return tf.reduce_mean(tf.reduce_max(matched, axis=1))

  img_sent_recall = [retrieval_recall(tf.transpose(sent_im_dist),
                                      tf.transpose(im_labels), k) for k in ks]
  sent_img_recall = [retrieval_recall(sent_im_dist, im_labels, k) for k in ks]

  return img_sent_recall + sent_img_recall
