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
import os

import tensorflow as tf
import time
from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

from attribute.utils import filter_infrequent_attribute
from data.common import EMB_DIR, MODEL_DIR
from data.bookcorpus import load_initialized_word_emb, load_data_with_attribute
from data.wiki103 import load_cross_domain_invert_data
from invert.models import MultiLabelInversionModel, MultiSetInversionModel, \
  RecurrentInversionModel
from invert.utils import sents_to_labels, count_label_freq, \
  tp_fp_fn_metrics, tp_fp_fn_metrics_np
from thought.quick_thought_model import QuickThoughtModel
from utils.common_utils import log
from utils.sent_utils import iterate_minibatches_indices, load_raw_sents, \
  load_embeds, get_similarity_metric

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags.DEFINE_integer('emb_dim', 620, 'embedding dimension')
flags.DEFINE_integer('encoder_dim', 1200, 'encoder dim')
flags.DEFINE_integer('context_size', 1, 'Context size')
flags.DEFINE_integer('num_layer', 3, 'Number of transformer layer')
flags.DEFINE_integer('batch_size', 800, 'Batch size')
flags.DEFINE_integer('epoch', 0, 'Epochs of training')
flags.DEFINE_integer('freq_min', 90,
                     'use rare words above this percentile, e.g. 80=the most '
                     'infrequent 20 percent sentences')
flags.DEFINE_integer('attack_batch_size', 128, 'Attack batch size')
flags.DEFINE_integer('max_iters', 1000, 'Max iterations for optimization')
flags.DEFINE_integer('seq_len', 10, 'Fixed recover sequence length')
flags.DEFINE_integer('train_size', 100, 'Number of authors data to use')
flags.DEFINE_integer('print_every', 1, 'Print metrics every iteration')

flags.DEFINE_float('C', 0.0, 'Label distribution aware margin')
flags.DEFINE_float('alpha', 0.0, 'Coefficient for regularization')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('wd', 1e-4, 'Weight decay')
flags.DEFINE_float('temp', 0.1, 'Temperature for optimization')

flags.DEFINE_string('cell_type', 'LSTM', 'Encoder model')
flags.DEFINE_string('model', 'multiset', 'Model for learning based inversion')
flags.DEFINE_string('opt', 'softmax', 'Optimization model')
flags.DEFINE_string('metric', 'l2', 'Metric to optimize')
flags.DEFINE_string('hub_model_name', 'quickthought', 'hub_model_name')
flags.DEFINE_string('model_dir',  os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('emb_dir',  EMB_DIR,
                    'Feature directory for saving embedding model')

flags.DEFINE_boolean('prior', False, 'Use deep prior for optimization')
flags.DEFINE_boolean('use_hub', False, 'Embedding from hub module')
flags.DEFINE_boolean('learning', False, 'Learning based inversion '
                                        'or optimize based')
flags.DEFINE_boolean('cross_domain', False, 'Cross domain data for learning '
                                            'based inversion')
FLAGS = flags.FLAGS


def load_inversion_data():
  all_filenames, attribute_dict = load_data_with_attribute('author')

  raw_dir = os.path.join(
    FLAGS.emb_dir, 'bookcorpus_raw_rare{}'.format(FLAGS.freq_min))
  sents, masks = load_raw_sents(all_filenames, raw_dir, rtn_filenames=False)

  model_type = FLAGS.cell_type
  if model_type == 'TRANS':
    model_type += 'l{}'.format(FLAGS.num_layer)

  model_name = 'e{}_{}_b{}'.format(FLAGS.epoch, model_type, FLAGS.batch_size)
  if FLAGS.use_hub:
    assert FLAGS.learning
    model_name = FLAGS.hub_model_name

  feat_dir = os.path.join(FLAGS.emb_dir, 'bookcorpus_{}_rare{}'.format(
    model_name, FLAGS.freq_min))
  log('Load embeddings from {}'.format(feat_dir))

  filenames, embs = load_embeds(all_filenames, feat_dir, rtn_filenames=True)
  assert len(embs) == len(sents)

  all_inputs = [embs, sents, masks]
  all_labels = np.asarray([attribute_dict[fname] for fname in filenames])

  filtered_indices = filter_infrequent_attribute(all_labels, 100)
  all_inputs = [inputs[filtered_indices] for inputs in all_inputs]

  all_embeds, all_sents, all_masks = all_inputs
  all_labels = all_labels[filtered_indices]
  all_attributes = np.sort(np.unique(all_labels))

  # split train and test
  train_attrs, test_attrs = train_test_split(all_attributes,
                                             train_size=FLAGS.train_size,
                                             test_size=FLAGS.train_size * 10,
                                             random_state=12345)

  train_attrs, test_attrs = set(train_attrs), set(test_attrs)
  train_indices, test_indices = [], []

  for idx, attr in enumerate(all_labels):
    if attr in train_attrs:
      train_indices.append(idx)
    if attr in test_attrs:
      test_indices.append(idx)

  if FLAGS.cross_domain:
    log('Load cross domain embeddings from {}'.format(FLAGS.hub_model_name))
    train_x, train_y, train_m = load_cross_domain_invert_data(
        FLAGS.hub_model_name)
  else:
    train_x, train_y, train_m = all_embeds[train_indices], all_sents[
      train_indices], all_masks[train_indices]

  test_x, test_y, test_m = all_embeds[test_indices], all_sents[test_indices], \
                           all_masks[test_indices]

  log('Loaded {} embeddings for inversion with shape {}'.format(
      test_x.shape[0], test_x.shape[1]))
  return train_x, test_x, train_y, test_y, train_m, test_m


def learning_invert(data, num_words, batch_size):
  train_x, test_x, train_y, test_y, train_m, test_m = data

  if FLAGS.model != 'rnn':
    train_y, test_y = sents_to_labels(train_y), sents_to_labels(test_y)

  label_freq = count_label_freq(train_y, num_words)
  log('Imbalace ratio: {}'.format(np.max(label_freq) / np.min(label_freq)))

  label_margin = tf.constant(np.reciprocal(label_freq ** 0.25),
                             dtype=tf.float32)
  C = FLAGS.C

  log('Build attack model for {} words...'.format(num_words))

  encoder_dim = train_x.shape[1]
  inputs = tf.placeholder(tf.float32, (None, encoder_dim), name="inputs")
  labels = tf.placeholder(tf.float32, (None, num_words), name="labels")
  masks = None
  training = tf.placeholder(tf.bool, name='training')

  if FLAGS.model == 'multiset':
    init_word_emb = load_initialized_word_emb()
    model = MultiSetInversionModel(init_word_emb.shape[1], num_words,
                                   FLAGS.seq_len, init_word_emb,
                                   C=C, label_margin=label_margin)
    preds, loss = model.forward(inputs, labels, training)
    true_pos, false_pos, false_neg = tp_fp_fn_metrics(labels, preds)
    eval_fetch = [loss, true_pos, false_pos, false_neg]
  elif FLAGS.model == 'rnn':
    labels = tf.placeholder(tf.int64, (None, None), name="labels")
    masks = tf.placeholder(tf.int32, (None, None), name="masks")

    init_word_emb = load_initialized_word_emb(glove_only=True)
    model = RecurrentInversionModel(init_word_emb.shape[1], num_words,
                                    FLAGS.seq_len, init_word_emb,
                                    beam_size=5, C=C, label_margin=label_margin)
    preds, loss = model.forward(inputs, labels, masks, training)
    eval_fetch = [loss, preds]
  elif FLAGS.model == 'multilabel':
    model = MultiLabelInversionModel(num_words, C=C, label_margin=label_margin)
    preds, loss = model.forward(inputs, labels, training)
    true_pos, false_pos, false_neg = tp_fp_fn_metrics(labels, preds)
    eval_fetch = [loss, true_pos, false_pos, false_neg]
  else:
    raise ValueError(FLAGS.model)

  t_vars = tf.trainable_variables()
  wd = FLAGS.wd
  post_ops = [tf.assign(v, v * (1 - wd)) for v in t_vars if 'kernel' in v.name]

  optimizer = tf.train.AdamOptimizer(FLAGS.lr)
  grads_and_vars = optimizer.compute_gradients(
    loss + tf.losses.get_regularization_loss(), t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars, global_step=tf.train.get_or_create_global_step())

  with tf.control_dependencies([train_ops]):
    train_ops = tf.group(*post_ops)

  log('Train attack model with {} data...'.format(len(train_x)))
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(50):
      train_iterations = 0
      train_loss = 0

      for batch_idx in iterate_minibatches_indices(len(train_y), batch_size,
                                                   True):
        if masks is None:
          one_hot_labels = np.zeros((len(batch_idx), num_words),
                                    dtype=np.float32)
          for i, idx in enumerate(batch_idx):
            one_hot_labels[i][train_y[idx]] = 1
          feed = {inputs: train_x[batch_idx], labels: one_hot_labels,
                  training: True}
        else:
          feed = {inputs: train_x[batch_idx], labels: train_y[batch_idx],
                  masks: train_m[batch_idx], training: True}

        err, _ = sess.run([loss, train_ops], feed_dict=feed)
        train_loss += err
        train_iterations += 1

      test_iterations = 0
      test_loss = 0
      test_tp, test_fp, test_fn = 0, 0, 0

      for batch_idx in iterate_minibatches_indices(len(test_y), batch_size=512,
                                                   shuffle=False):
        if masks is None:
          one_hot_labels = np.zeros((len(batch_idx), num_words),
                                    dtype=np.float32)
          for i, idx in enumerate(batch_idx):
            one_hot_labels[i][test_y[idx]] = 1
          feed = {inputs: test_x[batch_idx], labels: one_hot_labels,
                  training: False}
        else:
          feed = {inputs: test_x[batch_idx], labels: test_y[batch_idx],
                  masks: test_m[batch_idx], training: False}

        fetch = sess.run(eval_fetch, feed_dict=feed)
        if len(fetch) == 2:
          err, pred = fetch
          tp, fp, fn = tp_fp_fn_metrics_np(pred, test_y[batch_idx])
        else:
          err, tp, fp, fn = fetch

        test_iterations += 1
        test_loss += err
        test_tp += tp
        test_fp += fp
        test_fn += fn

      precision = test_tp / (test_tp + test_fp) * 100
      recall = test_tp / (test_tp + test_fn) * 100
      f1 = 2 * precision * recall / (precision + recall)

      log("Epoch: {}, train loss: {:.4f}, test loss: {:.4f}, "
          "pre: {:.2f}%, rec: {:.2f}%, f1: {:.2f}%".format(
            epoch, train_loss / train_iterations,
            test_loss / test_iterations,
            precision, recall, f1))


def optimization_invert(data, num_words, lr=1e-3, attack_batch_size=8,
                        seq_len=5, max_iters=1000):
  # use softmax to select words
  _, x, _, y = data[:4]
  y = sents_to_labels(y)

  model = QuickThoughtModel(num_words, FLAGS.emb_dim, FLAGS.encoder_dim,
                            FLAGS.context_size, FLAGS.cell_type,
                            num_layer=FLAGS.num_layer,
                            init_word_emb=None, train=False)
  model_type = FLAGS.cell_type
  if model_type == 'TRANS':
    model_type += 'l{}'.format(FLAGS.num_layer)

  model_name = 'e{}_{}_b{}'.format(FLAGS.epoch, model_type, FLAGS.batch_size)
  word_emb = model.word_in_emb

  if FLAGS.prior:
    attack_batch_size = 1  # one prior NN per example
    init_word_emb = load_initialized_word_emb()
    prior_encoder_dim = init_word_emb.shape[1]

    inputs = tf.get_variable(
      name='inputs', shape=(attack_batch_size, seq_len, prior_encoder_dim),
      initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=False)

    rnn_prior = tf.keras.layers.CuDNNLSTM(init_word_emb.shape[1],
                                          return_sequences=True,
                                          name='prior_rnn')
    prior_word_emb = tf.Variable(
      tf.convert_to_tensor(init_word_emb, dtype=tf.float32),
      name='prior_word_emb')

    rnn_outputs = rnn_prior(inputs)
    logit_inputs = tf.matmul(rnn_outputs, prior_word_emb, transpose_b=True)

    t_vars = [v for v in tf.trainable_variables() if v.name.startswith('prior')]
    t_var_names = set(v.name for v in t_vars + [inputs])
  else:
    logit_inputs = tf.get_variable(
      name='inputs', shape=(attack_batch_size, seq_len, num_words),
      initializer=tf.random_uniform_initializer(-0.1, 0.1))

    t_vars = [logit_inputs]
    t_var_names = {logit_inputs.name}

  prob_inputs = tf.nn.softmax(logit_inputs / FLAGS.temp, axis=-1)
  masks = tf.ones(shape=(attack_batch_size, seq_len), dtype=tf.int32)
  targets = tf.placeholder(tf.float32,
                           shape=(attack_batch_size, FLAGS.encoder_dim))

  emb_inputs = tf.matmul(prob_inputs, word_emb)
  encoded = model.encode(emb_inputs, masks, model.in_cells, model.proj_in)
  loss = get_similarity_metric(encoded, targets, FLAGS.metric, rtn_loss=True)
  loss = tf.reduce_sum(loss)
  if FLAGS.alpha > 0.:
    # encourage the words to be different
    diff = tf.expand_dims(prob_inputs, 2) - tf.expand_dims(prob_inputs, 1)
    reg = tf.reduce_mean(-tf.exp(tf.reduce_sum(diff ** 2, axis=-1)), [1, 2])
    loss += FLAGS.alpha * tf.reduce_sum(reg)

  optimizer = tf.train.AdamOptimizer(lr)
  model_vars = [v for v in tf.global_variables() if v.name not in t_var_names]
  saver = tf.train.Saver(model_vars)
  start_vars = set(v.name for v in model_vars)

  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars, global_step=tf.train.get_or_create_global_step())
  end_vars = tf.global_variables()
  new_vars = [v for v in end_vars if v.name not in start_vars]

  preds = tf.argmax(prob_inputs, axis=-1)
  batch_init_ops = tf.variables_initializer(new_vars)

  total_it = len(x) // attack_batch_size
  with tf.Session() as sess:
    saver.restore(sess, os.path.join(FLAGS.model_dir,
                                     'bookcorpus_{}'.format(model_name),
                                     'model.ckpt'))

    def invert_one_batch(batch_targets):
      sess.run(batch_init_ops)
      for i in range(max_iters):
        sess.run([loss, train_ops], feed_dict={targets: batch_targets})
      return sess.run([preds, loss], feed_dict={targets: batch_targets})

    eval_opt_invert(x, y, invert_one_batch, attack_batch_size, total_it)


def eval_opt_invert(x, y, invert_opt_fn, attack_batch_size, total_it):
  it = 0.0
  all_tp, all_fp, all_fn, all_err = 0.0, 0.0, 0.0, 0.0

  start_time = time.time()
  for batch_idx in iterate_minibatches_indices(len(x), attack_batch_size, False,
                                               False):
    y_pred, err = invert_opt_fn(x[batch_idx])
    tp, fp, fn = tp_fp_fn_metrics_np(y_pred, y[batch_idx])

    it += 1.0
    all_err += err
    all_tp += tp
    all_fp += fp
    all_fn += fn

    all_pre = all_tp / (all_tp + all_fp + 1e-7)
    all_rec = all_tp / (all_tp + all_fn + 1e-7)
    all_f1 = 2 * all_pre * all_rec / (all_pre + all_rec + 1e-7)

    if it % FLAGS.print_every == 0:
      it_time = (time.time() - start_time) / it
      log("Iter {:.2f}%, err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%,"
          " {:.2f} sec/it".format(it / total_it * 100, all_err / it,
                                  all_pre * 100, all_rec * 100,
                                  all_f1 * 100, it_time))

  log("Final err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%".format(
    all_err / it, all_pre * 100, all_rec * 100, all_f1 * 100))


def main(unused_argv):
  data = load_inversion_data()
  if FLAGS.learning:
    learning_invert(data=data, num_words=50001,
                    batch_size=FLAGS.attack_batch_size)
  else:
    optimization_invert(data=data, num_words=50001, seq_len=FLAGS.seq_len,
                        attack_batch_size=FLAGS.attack_batch_size,
                        max_iters=FLAGS.max_iters)


if __name__ == '__main__':
  app.run(main)
