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
from tensorflow.python.util import nest
from tensor2tensor.utils import beam_search


class MultiLabelInversionModel(object):
  def __init__(self, output_size, C=0., label_margin=None, init_word_emb=None,
               drop_p=0.25):
    self.C = C
    self.label_margin = label_margin
    self.dropout = tf.keras.layers.Dropout(drop_p)
    self.fc = tf.keras.layers.Dense(output_size, name='output')
    if init_word_emb is not None:
      self.embedding = tf.convert_to_tensor(init_word_emb, dtype=tf.float32)
    else:
      self.embedding = None

  def forward(self, inputs, labels, training):
    x = self.dropout.apply(inputs, training=training)
    logits = self.fc.apply(x)
    return self.predict(logits, labels)

  def forward_with_prefix(self, inputs, prefixes, labels, training):
    x = self.dropout.apply(inputs, training=training)
    prefixes_embeds = tf.nn.embedding_lookup(self.embedding, prefixes)
    x = tf.concat([x, tf.reduce_mean(prefixes_embeds, axis=1)], axis=1)
    logits = self.fc.apply(x)
    return self.predict(logits, labels)

  def predict(self, logits, labels):
    logits_train = logits - self.C * self.label_margin * labels \
      if self.C > 0 else logits

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                   logits=logits_train)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    preds = tf.round(tf.nn.sigmoid(logits))
    return preds, loss


class RecurrentInversionModel(object):
  def __init__(self, emb_dim, output_size, seq_len=10, init_word_emb=None,
               beam_size=5, C=0., label_margin=None, drop_p=0.25):
    self.C = C
    self.label_margin = label_margin
    self.beam_size = beam_size
    self.seq_len = seq_len
    self.emb_dim = emb_dim
    self.output_size = output_size
    self.decoder = tf.keras.layers.CuDNNLSTM(emb_dim, return_sequences=True)
    self.dropout = tf.keras.layers.Dropout(drop_p)
    self.fc = tf.keras.layers.Dense(emb_dim, use_bias=False)

    self.eos_id = 0
    self.sos_id = output_size
    self.embedding = tf.Variable(self.get_or_init_word_emb(init_word_emb),
                                 name="output_emb")

  def get_or_init_word_emb(self, weights=None):
    if weights is None:
      # add output size 1 for EOS used in beam search
      return tf.random_uniform((self.output_size + 1, self.emb_dim), -0.1, 0.1)
    else:
      # add eos
      weights = np.vstack(
        [weights, np.zeros((1, weights.shape[1]), dtype=np.float32)])
      return tf.convert_to_tensor(weights, dtype=tf.float32)

  def forward(self, inputs, labels, masks, training):
    b = tf.shape(labels)[0]
    label_inputs = tf.concat([tf.zeros((b, 1), tf.int64), labels[:, :-1]],
                             axis=1)
    mask_inputs = tf.concat([tf.ones((b, 1), tf.int32), masks[:, :-1]], axis=1)

    emb_inputs = tf.nn.embedding_lookup(self.embedding, label_inputs)
    rnn_masks = tf.cast(tf.expand_dims(mask_inputs, 2), tf.float32)
    rnn_inputs = tf.multiply(emb_inputs, rnn_masks)

    self.dropout.apply(rnn_inputs, training=training)

    h0 = self.fc.apply(inputs)
    rnn_outputs = self.decoder.apply(rnn_inputs, initial_state=[h0, h0])
    self.dropout.apply(rnn_outputs, training=training)

    logits = tf.matmul(rnn_outputs, self.embedding, transpose_b=True)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                          logits=logits)
    loss = tf.reduce_sum(loss * tf.cast(masks, tf.float32), axis=-1)
    loss = tf.reduce_mean(loss)

    def pred_fn(x, i, states):
      e = tf.nn.embedding_lookup(self.embedding, x)
      r = self.decoder.apply(e, initial_state=states)[:, -1, :]
      o = tf.matmul(r, self.embedding, transpose_b=True)
      return o, states

    initial_ids = tf.ones((b,), tf.int32) * self.sos_id
    beam_preds, _, _ = beam_search.beam_search(
      pred_fn, initial_ids, alpha=0., beam_size=self.beam_size,
      decode_length=self.seq_len, vocab_size=self.output_size + 1,
      eos_id=self.eos_id, states=[h0, h0])

    return beam_preds[:, 0, 1:], loss


class MultiSetInversionModel(object):
  def __init__(self, emb_dim, output_size, steps=5, init_word_emb=None, C=0.,
               label_margin=None, unrolled=False, drop_p=0.25):
    self.C = C
    self.label_margin = label_margin
    self.steps = steps
    self.emb_dim = emb_dim
    self.output_size = output_size
    self.eos_id = 0
    self.fc = tf.keras.layers.Dense(emb_dim, use_bias=False)
    self.policy = tf.keras.layers.LSTMCell(emb_dim)
    self.dropout = tf.keras.layers.Dropout(drop_p)
    self.embedding = tf.Variable(self.get_or_init_word_emb(init_word_emb),
                                 name="output_emb")
    self.unrolled = unrolled

  def get_or_init_word_emb(self, weights=None):
    if weights is None:
      return tf.random_uniform((self.output_size, self.emb_dim), -0.1, 0.1)
    else:
      return tf.convert_to_tensor(weights, dtype=tf.float32)

  def forward(self, inputs, labels, training):
    states = self.policy.get_initial_state(inputs)
    xt = inputs
    xt = self.fc.apply(xt)
    if self.unrolled:
      return self.unrolled_predict_loop(labels, xt, states, training)
    else:
      return self.predict_loop(labels, xt, states, training)

  def forward_with_prefix(self, inputs, prefixes, labels, training):
    xt = inputs
    xt = self.fc.apply(xt)

    prefixes_embeds = tf.nn.embedding_lookup(self.embedding, prefixes)
    prefixes_embeds = tf.reduce_mean(prefixes_embeds, axis=1)
    states = [prefixes_embeds, prefixes_embeds]
    if self.unrolled:
      return self.unrolled_predict_loop(labels, xt, states, training)
    else:
      return self.predict_loop(labels, xt, states, training)

  def unrolled_predict_loop(self, labels, xt, states, training):
    batch_size = tf.shape(labels)[0]
    labels_t = tf.identity(labels)
    finished_t = tf.zeros(batch_size, dtype=tf.bool)     # B

    predictions = []
    losses = []
    for t in range(self.steps):
      xt = self.dropout.apply(xt, training=training)

      ht, states = self.policy.apply(xt, states)
      logits = tf.matmul(ht, self.embedding, transpose_b=True)

      yt = tf.argmax(logits, axis=1)  # predicted label at this step
      yt = tf.where(finished_t, self.eos_id * tf.ones_like(yt), yt)

      # update finished
      finished = tf.equal(yt, self.eos_id)
      finished_t = tf.logical_or(finished_t, finished)

      xt = tf.nn.embedding_lookup(self.embedding, yt)
      yt_one_hot = tf.one_hot(yt, self.output_size)
      predictions.append(tf.cast(yt_one_hot, tf.bool))

      # labels left to predict y_true & (~y_pred)
      labels_t = tf.logical_and(tf.cast(labels_t, tf.bool),
                                tf.logical_not(tf.cast(yt_one_hot, tf.bool)))
      labels_t = tf.cast(labels_t, tf.float32)

      logits_train = logits - self.C * self.label_margin * labels_t \
          if self.C > 0 else logits
      loss_t = -tf.nn.log_softmax(logits_train) * labels_t
      losses.append(tf.reduce_sum(loss_t, axis=1))

    preds = tf.cast(tf.reduce_any(predictions, axis=0),
                    tf.float32)  # union the predictions
    loss = tf.reduce_mean(losses)
    return preds, loss

  def predict_loop(self, labels, xt, states, training):
    # B = batch size, V = vocab size, E = embedding size
    batch_size = tf.shape(labels)[0]
    init_labels_t = tf.identity(labels)  # B x V
    init_input_t = xt       # B x E
    init_states_t = states  # B x E
    init_finished_t = tf.zeros(batch_size, dtype=tf.bool)     # B
    init_prediction_t = tf.zeros_like(labels, dtype=tf.bool)  # B x V
    init_loss_t = tf.zeros(batch_size)

    def _is_not_finished(i, labels_t, input_t, states_t, finished_t,
                         prediction_t, loss_t):
      return tf.less(i, self.steps)

    def _inner_loop(i, labels_t, input_t, states_t, finished_t, prediction_t,
                    loss_t):
      input_t = self.dropout(input_t, training=training)
      ht, states_t = self.policy.apply(input_t, states_t)
      logits = tf.matmul(ht, self.embedding, transpose_b=True)
      # predicted label at this step
      yt = tf.argmax(logits, axis=1)
      # if finished, set prediction to EOS id
      yt = tf.where(finished_t, self.eos_id * tf.ones_like(yt), yt)
      # update finished
      finished = tf.equal(yt, self.eos_id)
      finished_t = tf.logical_or(finished_t, finished)
      # inputs for next iteration of loop
      input_t = tf.nn.embedding_lookup(self.embedding, yt)
      yt_one_hot = tf.one_hot(yt, self.output_size)
      prediction_t = tf.logical_or(tf.cast(yt_one_hot, tf.bool),
                                   prediction_t)
      # labels left to predict y_true & (~y_pred)
      labels_t = tf.logical_and(tf.cast(labels_t, tf.bool),
                                tf.logical_not(tf.cast(yt_one_hot, tf.bool)))
      labels_t = tf.cast(labels_t, tf.float32)

      logits_train = logits - self.C * self.label_margin * labels_t \
          if self.C > 0 else logits

      # loss is the sum of words that needs to be predicted
      loss = -tf.nn.log_softmax(logits_train) * labels_t
      loss = tf.reduce_sum(loss, axis=1)  # B
      loss_t += loss
      return i + 1, labels_t, input_t, states_t, finished_t, prediction_t, \
          loss_t

    shape_invariants = [
        tf.TensorShape([]),
        init_labels_t.get_shape(),
        init_input_t.get_shape(),
        nest.map_structure(lambda state: state.get_shape(), init_states_t),
        init_finished_t.get_shape(),
        init_prediction_t.get_shape(),
        init_loss_t.get_shape()
    ]

    _, _, _, _, final_finished, final_prediction, final_loss = tf.while_loop(
      _is_not_finished,
      _inner_loop,
      loop_vars=[tf.constant(0), init_labels_t, init_input_t, init_states_t,
                 init_finished_t, init_prediction_t, init_loss_t],
      shape_invariants=shape_invariants,
    )

    final_loss = tf.reduce_mean(final_loss / tf.cast(self.steps, tf.float32))

    return final_prediction, final_loss
