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

from data.bookcorpus import load_initialized_word_emb, build_vocabulary, \
  load_author_data as load_bookcorpus_author
from data.reddit import load_author_data as reddit_author_data
from data.common import MODEL_DIR
from data.wiki103 import load_wiki_cross_domain_data as load_cross_domain_data
from invert.models import MultiLabelInversionModel, MultiSetInversionModel, \
  RecurrentInversionModel
from invert.utils import sents_to_labels, count_label_freq, \
  tp_fp_fn_metrics, tp_fp_fn_metrics_np, sinkhorn, continuous_topk_v2
from text_encoder import encode_sentences
from train_feature_mapper import linear_mapping, mlp_mapping, gan_mapping
from thought import get_model_ckpt_name, get_model_config
from thought.quick_thought_model import QuickThoughtModel
from utils.common_utils import log
from utils.sent_utils import iterate_minibatches_indices, get_similarity_metric

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags.DEFINE_integer('epoch', 0, 'Epochs of training')
flags.DEFINE_integer('batch_size', 800, 'Batch size')
flags.DEFINE_integer('attack_batch_size', 32, 'Attack batch size')
flags.DEFINE_integer('max_iters', 1000, 'Max iterations for optimization')
flags.DEFINE_integer('seq_len', 15, 'Fixed recover sequence length')
flags.DEFINE_integer('train_size', 250, 'Number of authors data to use')
flags.DEFINE_integer('test_size', 125, 'Number of authors data to test')
flags.DEFINE_integer('print_every', 1, 'Print metrics every iteration')
flags.DEFINE_integer('high_layer_idx', -1, 'Output layer index')
flags.DEFINE_integer('low_layer_idx', -1, 'Optimize layer index')
flags.DEFINE_float('C', 0.0, 'Label distribution aware margin')
flags.DEFINE_float('alpha', 0.0, 'Coefficient for regularization')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('wd', 1e-4, 'Weight decay')
flags.DEFINE_float('temp', 0.1, 'Temperature for optimization')
flags.DEFINE_float('gamma', 0.0, 'Loss ratio for adversarial')
flags.DEFINE_string('attr', 'word', 'Attributes to censor')
flags.DEFINE_string('model', 'multiset', 'Model for learning based inversion')
flags.DEFINE_string('metric', 'l2', 'Metric to optimize')
flags.DEFINE_string('model_name', 'quickthought', 'Model name')
flags.DEFINE_string('model_dir', os.path.join(MODEL_DIR, 'models', 's2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('data_name', 'bookcorpus', 'Data name')
flags.DEFINE_string('mapper', 'linear', 'Mapper to use')
flags.DEFINE_boolean('permute', False, 'Optimize permutation matrix')
flags.DEFINE_boolean('learning', False, 'Learning based inversion '
                                        'or optimize based')
flags.DEFINE_boolean('cross_domain', False, 'Cross domain data for learning '
                                            'based inversion')
FLAGS = flags.FLAGS


def load_inversion_data():
  vocab = build_vocabulary(rebuild=False)

  if FLAGS.data_name == 'bookcorpus':
    train_sents, _, test_sents, _, _, _ = load_bookcorpus_author(
        train_size=FLAGS.train_size, test_size=FLAGS.test_size,
        unlabeled_size=0, split_by_book=True, split_word=True,
        top_attr=800)
  elif FLAGS.data_name == 'reddit':
    train_sents, _, test_sents, _, _, _ = reddit_author_data(
        train_size=FLAGS.train_size, test_size=1, unlabeled_size=0,
        split_word=True, top_attr=0)
  else:
    raise ValueError(FLAGS.data_name)

  if FLAGS.cross_domain:
    train_sents = load_cross_domain_data(800000, split_word=True)
    log('Loaded {} cross domain sentences'.format(len(train_sents)))

  ckpt_name = get_model_ckpt_name(FLAGS.model_name, epoch=FLAGS.epoch,
                                  batch_size=FLAGS.batch_size,
                                  gamma=FLAGS.gamma, num_layer=3,
                                  attr=FLAGS.attr)
  model_path = os.path.join(FLAGS.model_dir, ckpt_name, 'model.ckpt')
  config = get_model_config(FLAGS.model_name)

  train_data, test_data = encode_sentences(
      vocab, model_path, config, train_sents, test_sents,
      low_layer_idx=FLAGS.low_layer_idx, high_layer_idx=FLAGS.high_layer_idx)
  # clear session data for later optimization or learning
  tf.keras.backend.clear_session()

  train_x, train_y, train_m = train_data
  test_x, test_y, test_m = test_data

  if FLAGS.low_layer_idx != FLAGS.high_layer_idx:
    log('Training high to low mapping...')
    if FLAGS.mapper == 'linear':
      mapping = linear_mapping(train_x[1], train_x[0])
    elif FLAGS.mapper == 'mlp':
      mapping = mlp_mapping(train_x[1], train_x[0], epochs=50,
                            activation=tf.nn.relu)
    elif FLAGS.mapper == 'gan':
      mapping = gan_mapping(train_x[1], train_x[0], disc_iters=5,
                            batch_size=64, gamma=1.0, epoch=100,
                            activation=tf.tanh)
    else:
      raise ValueError(FLAGS.mapper)
    test_x = mapping(test_x[1])
    train_x = train_x[0]

  log('Loaded {} embeddings for inversion with shape {}'.format(
      test_x.shape[0], test_x.shape[1]))

  data = (train_x, test_x, train_y, test_y, train_m, test_m)
  return data


def learning_invert(data, batch_size):
  train_x, test_x, train_y, test_y, train_m, test_m = data

  config = get_model_config(FLAGS.model_name)
  num_words = config['vocab_size']

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
    if num_words == 50001:
      init_word_emb = load_initialized_word_emb()
      emb_dim = init_word_emb.shape[1]
    else:
      init_word_emb = None
      emb_dim = 512

    model = MultiSetInversionModel(emb_dim, num_words,
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

    for epoch in range(30):
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

        # for yy, pp in zip(test_y[batch_idx], pred):
        #   matched = np.intersect1d(np.unique(yy), np.unique(pp))
        #   if len(matched) >= 0.8 * len(yy):
        #     print(' '.join([inv_vocab[w] for w in yy]))
        #     print(' '.join([inv_vocab[w] for w in np.unique(pp)]))

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


def optimization_invert(data, lr=1e-3, attack_batch_size=8,
                        seq_len=5, max_iters=1000):
  # use softmax to select words
  _, x, _, y = data[:4]
  y = sents_to_labels(y)

  config = get_model_config(FLAGS.model_name)
  num_words = config['vocab_size']
  model = QuickThoughtModel(num_words, config['emb_dim'],
                            config['encoder_dim'], 1, init_word_emb=None,
                            cell_type=config['cell_type'],
                            num_layer=config['num_layer'], train=False)
  word_emb = model.word_in_emb
  targets = tf.placeholder(tf.float32,
                           shape=(attack_batch_size, x.shape[1]))

  log('Inverting {} words from {} embeddings'.format(num_words, len(x)))

  if FLAGS.permute:
    # modeling the top k words then permute the order
    logit_inputs = tf.get_variable(
      name='inputs',
      shape=(attack_batch_size, seq_len, num_words - 1),
      initializer=tf.random_uniform_initializer(-0.1, 0.1))
    t_vars = [logit_inputs]

    prob_inputs = continuous_topk_v2(logit_inputs, seq_len, FLAGS.temp)
    pad_inputs = tf.zeros((attack_batch_size, seq_len, 1))
    prob_inputs = tf.concat([pad_inputs, prob_inputs], axis=2)
    emb_inputs = tf.matmul(prob_inputs, word_emb)

    permute_inputs = tf.get_variable(
        name='permute_inputs',
        shape=(attack_batch_size, seq_len, seq_len),
        initializer=tf.truncated_normal_initializer(0, 0.1))
    t_vars.append(permute_inputs)

    permute_matrix = sinkhorn(permute_inputs / FLAGS.temp, 20)
    emb_inputs = tf.matmul(permute_matrix, emb_inputs)
  else:
    logit_inputs = tf.get_variable(
      name='inputs',
      shape=(attack_batch_size, seq_len, num_words - 1),
      initializer=tf.random_uniform_initializer(-0.1, 0.1))
    t_vars = [logit_inputs]

    pad_inputs = tf.ones((attack_batch_size, seq_len, 1), tf.float32) * (-1e9)
    logit_inputs = tf.concat([pad_inputs, logit_inputs], axis=2)
    prob_inputs = tf.nn.softmax(logit_inputs / FLAGS.temp, axis=-1)
    emb_inputs = tf.matmul(prob_inputs, word_emb)

  preds = tf.argmax(prob_inputs, axis=-1)
  t_var_names = set([v.name for v in t_vars])

  masks = tf.ones(shape=(attack_batch_size, seq_len), dtype=tf.int32)
  all_layers = model.encode(emb_inputs, masks, model.in_cells, model.proj_in,
                            return_all_layers=True)
  encoded = all_layers[FLAGS.low_layer_idx]

  loss = get_similarity_metric(encoded, targets, FLAGS.metric, rtn_loss=True)
  loss = tf.reduce_sum(loss)

  if FLAGS.alpha > 0.:
    # encourage the words to be different
    diff = tf.expand_dims(prob_inputs, 2) - tf.expand_dims(prob_inputs, 1)
    reg = tf.reduce_mean(-tf.exp(tf.reduce_sum(diff ** 2, axis=-1)), [1, 2])
    loss += FLAGS.alpha * tf.reduce_sum(reg)

  optimizer = tf.train.AdamOptimizer(lr)
  model_vars = [v for v in tf.global_variables()
                if v.name not in t_var_names]
  saver = tf.train.Saver(model_vars)
  start_vars = set(v.name for v in model_vars)

  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
      grads_and_vars, global_step=tf.train.get_or_create_global_step())
  end_vars = tf.global_variables()
  new_vars = [v for v in end_vars if v.name not in start_vars]

  batch_init_ops = tf.variables_initializer(new_vars)

  total_it = len(x) // attack_batch_size
  with tf.Session() as sess:
    ckpt_name = get_model_ckpt_name(FLAGS.model_name, epoch=FLAGS.epoch,
                                    batch_size=FLAGS.batch_size, num_layer=3,
                                    gamma=FLAGS.gamma, attr=FLAGS.attr)
    ckpt_path = os.path.join(FLAGS.model_dir, ckpt_name, 'model.ckpt')
    log('Restoring model from {}'.format(ckpt_path))
    saver.restore(sess, ckpt_path)

    def invert_one_batch(batch_targets):
      sess.run(batch_init_ops)
      feed_dict = {targets: batch_targets}
      prev = 1e6
      for i in range(max_iters):
        curr, _ = sess.run([loss, train_ops], feed_dict)
        # stop if no progress
        if (i + 1) % (max_iters // 10) == 0 and curr > prev:
          break
        prev = curr
      return sess.run([preds, loss], feed_dict)

    it = 0.0
    all_tp, all_fp, all_fn, all_err = 0.0, 0.0, 0.0, 0.0

    start_time = time.time()

    # vocab = build_vocabulary(exp_id=0, rebuild=False)
    # inv_vocab = dict((v, k) for k, v in vocab.items())

    for batch_idx in iterate_minibatches_indices(len(x), attack_batch_size,
                                                 False, False):
      y_pred, err = invert_one_batch(x[batch_idx])
      tp, fp, fn = tp_fp_fn_metrics_np(y_pred, y[batch_idx])
      # for yy, pp in zip(y[batch_idx], y_pred):
      #   matched = np.intersect1d(np.unique(yy), np.unique(pp))
      #   if len(matched) >= 0.75 * len(yy):
      #     print(' '.join([inv_vocab[w] for w in yy]))
      #     print(' '.join([inv_vocab[w] for w in np.unique(pp)]))

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
        log('Iter {:.2f}%, err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%,'
            ' {:.2f} sec/it'.format(it / total_it * 100, all_err / it,
                                    all_pre * 100, all_rec * 100, all_f1 * 100,
                                    it_time))

    all_pre = all_tp / (all_tp + all_fp + 1e-7)
    all_rec = all_tp / (all_tp + all_fn + 1e-7)
    all_f1 = 2 * all_pre * all_rec / (all_pre + all_rec + 1e-7)
    log('Final err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%'.format(
        all_err / it, all_pre * 100, all_rec * 100, all_f1 * 100))


def main(_):
  data = load_inversion_data()
  if FLAGS.learning:
    learning_invert(data=data, batch_size=FLAGS.attack_batch_size)
  else:
    optimization_invert(data=data, seq_len=FLAGS.seq_len,
                        attack_batch_size=FLAGS.attack_batch_size,
                        max_iters=FLAGS.max_iters)


if __name__ == '__main__':
  app.run(main)
