from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import tensorflow as tf
from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

from attribute.mixmatch import guess_label
from attribute.utils import batch_interpolation
from data.common import EMB_DIR, MODEL_DIR
from data.bookcorpus import split_bookcorpus
from membership.models import BilinearMetricModel
from membership.utils import compute_adversarial_advantage, \
  adversarial_advantage_from_trained
from thought import ThoughtModelNameFunc
from utils.common_utils import log
from utils.sent_utils import iterate_minibatches_indices, inf_batch_iterator, \
  load_embeds

tf.logging.set_verbosity(tf.logging.FATAL)

flags.DEFINE_integer('epoch', 0, 'Epochs of training')
flags.DEFINE_integer('freq_min', 90,
                     'use rare words above this percentile, e.g. 80=the most '
                     'infrequent 20 percent sentences')
flags.DEFINE_integer('k', 1, 'Number of augmentations')
flags.DEFINE_integer('encoder_dim', 1200, 'Encoder dimension')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('n_book', 4000, 'Number of books to test')
flags.DEFINE_float('train_size', 0.1, 'Ratio of data for training the '
                                      'membership inference attack model')
flags.DEFINE_float('temp', 0.5, 'Temp for sharpening')
flags.DEFINE_float('lambda_u', 0.0, 'Lambda for unlabeled data loss')
flags.DEFINE_string('metric', 'dot', 'Similarity metric')
flags.DEFINE_string('emb_dir',  EMB_DIR,
                    'Feature directory for saving embedding model')
flags.DEFINE_string('model_dir',  os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('model_name', 'quickthought', 'Model name')
flags.DEFINE_boolean('book_level', False, 'Evaluate on book level')

FLAGS = flags.FLAGS


def merge_context(embs_a, embs_b):
  merged = []
  for a, b in zip(embs_a, embs_b):
    merged.append(np.hstack([a, b]))
  return merged


def load_text_embedding(train_filenames, test_filenames, freq_min, merge=True):
  model_name = ThoughtModelNameFunc[FLAGS.model_name](FLAGS.epoch)
  train_filenames = train_filenames[:FLAGS.n_book]
  test_filenames = test_filenames[:FLAGS.n_book]
  feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}'.format(
    model_name, freq_min))
  ctx_feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}_ctx'.format(
    model_name, freq_min))

  print('Loading embeddings from {}...'.format(feat_dir))

  member_embeds_a = load_embeds(train_filenames, feat_dir, stack=not merge)
  member_embeds_b = load_embeds(train_filenames, ctx_feat_dir, stack=not merge)
  assert len(member_embeds_a) == len(member_embeds_b)
  if merge:
    member_embeds = merge_context(member_embeds_a, member_embeds_b)
    del member_embeds_a, member_embeds_b
  else:
    member_embeds = [member_embeds_a, member_embeds_b]

  nonmember_embeds_a = load_embeds(test_filenames, feat_dir,
                                   stack=not merge)
  nonmember_embeds_b = load_embeds(test_filenames, ctx_feat_dir,
                                   stack=not merge)
  assert len(nonmember_embeds_a) == len(nonmember_embeds_b)
  if merge:
    nonmember_embeds = merge_context(nonmember_embeds_a, nonmember_embeds_b)
    del nonmember_embeds_a, nonmember_embeds_b
  else:
    nonmember_embeds = [nonmember_embeds_a, nonmember_embeds_b]

  return member_embeds, nonmember_embeds


def collect_scores(inputs, embs, labels, sess, fetch):
  stacked_embs = np.vstack(embs)
  stacked_scores = []
  encoder_dim = stacked_embs.shape[1] // 2
  for batch_idx in iterate_minibatches_indices(
    len(stacked_embs), batch_size=2048, shuffle=False):

    if isinstance(inputs, list):
      feed = {inputs[0]: stacked_embs[batch_idx][:, :encoder_dim],
              inputs[1]: stacked_embs[batch_idx][:, encoder_dim:]}
    else:
      feed = {inputs: stacked_embs[batch_idx]}

    scores = sess.run(fetch, feed_dict=feed)
    stacked_scores.append(scores)
  stacked_scores = np.concatenate(stacked_scores)

  member_metrics, nonmember_metrics = [], []
  start_idx = 0
  for emb, label in zip(embs, labels):
    scores = stacked_scores[start_idx: start_idx + len(emb)]
    start_idx += len(emb)
    if label == 1:
      member_metrics.append(scores)
    else:
      nonmember_metrics.append(scores)
  return member_metrics, nonmember_metrics


def trained_metric_attack():
  freq_min = FLAGS.freq_min
  train_filenames, test_filenames = split_bookcorpus(0)
  encoder_dim = FLAGS.encoder_dim
  # model = LinearMetricModel(encoder_dim // 2)
  model = BilinearMetricModel(encoder_dim)
  optimizer = tf.train.AdamOptimizer(1e-4)
  inputs_a = tf.placeholder(tf.float32, (None, encoder_dim), name="inputs_a")
  inputs_b = tf.placeholder(tf.float32, (None, encoder_dim), name="inputs_b")
  labels = tf.placeholder(tf.float32, (None,), name="labels")

  logits = model.forward(inputs_a, inputs_b)

  if FLAGS.metric == 'dot':
    sim = tf.reduce_sum(tf.multiply(inputs_a, inputs_b), axis=1)
  elif FLAGS.metric == 'cosine':
    sim = tf.reduce_sum(
      tf.multiply(tf.nn.l2_normalize(inputs_a, axis=-1),
                  tf.nn.l2_normalize(inputs_b, axis=-1)), axis=1)
  elif FLAGS.metric == 'l2':
    sim = - tf.reduce_sum(tf.square(inputs_a - inputs_b), axis=1)
  else:
    raise ValueError(FLAGS.metric)

  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.reduce_mean(loss)

  t_vars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars,  global_step=tf.train.get_or_create_global_step())

  member_embeds, nonmember_embeds = load_text_embedding(
    train_filenames, test_filenames, freq_min)

  membership_labels = np.concatenate(
    [np.ones(len(member_embeds)), np.zeros(len(nonmember_embeds))])

  train_embeds, test_embeds, train_labels, test_labels = train_test_split(
    member_embeds + nonmember_embeds, membership_labels,
    train_size=FLAGS.train_size, stratify=membership_labels)

  train_y = []
  for emb, label in zip(train_embeds, train_labels):
    train_y.append(np.ones(len(emb)) * label)

  train_y = np.concatenate(train_y).astype(np.float32)
  train_x = np.vstack(train_embeds)

  inputs = [inputs_a, inputs_b]
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    def calculate_adversarial_advantage(baseline):
      fetch = sim if baseline else logits
      test_member_metrics, test_nonmember_metrics = collect_scores(
        inputs, test_embeds, test_labels, sess, fetch)

      compute_adversarial_advantage(np.concatenate(test_member_metrics),
                                    np.concatenate(test_nonmember_metrics))
      if FLAGS.book_level:
        train_member_metrics, train_nonmember_metrics = collect_scores(
          inputs, train_embeds, train_labels, sess, fetch)
        all_metrics_flat = np.concatenate(
          train_member_metrics + train_nonmember_metrics
          + test_member_metrics + test_nonmember_metrics)
        metric_range = (np.min(all_metrics_flat), np.max(all_metrics_flat))

        data = (train_member_metrics, train_nonmember_metrics,
                test_member_metrics, test_nonmember_metrics)
        adversarial_advantage_from_trained(data, metric_range)

    calculate_adversarial_advantage(True)
    print('Training attack model...')
    for epoch in range(10):
      iterations = 0
      train_loss = 0

      for batch_idx in iterate_minibatches_indices(
        len(train_y), batch_size=FLAGS.batch_size, shuffle=True):
        feed = {inputs_a: train_x[batch_idx][:, :encoder_dim],
                inputs_b: train_x[batch_idx][:, encoder_dim:],
                labels: train_y[batch_idx]}
        err, _ = sess.run([loss, train_ops], feed_dict=feed)
        train_loss += err
        iterations += 1

      log("\nEpoch: {}, Loss: {:.4f}".format(epoch, train_loss / iterations))
      calculate_adversarial_advantage(False)


def train_semisupervised(k=2, temp=0.5, lambda_u=10.0):
  batch_size = FLAGS.batch_size
  freq_min = FLAGS.freq_min

  train_filenames, test_filenames = split_bookcorpus(0)
  encoder_dim = FLAGS.encoder_dim
  model = BilinearMetricModel(FLAGS.encoder_dim)

  optimizer = tf.train.AdamOptimizer(1e-4)
  inputs = tf.placeholder(tf.float32, (None, encoder_dim * 2),
                          name="inputs")
  u_inputs = tf.placeholder(tf.float32, (None, encoder_dim * 2),
                            name="unlabeled_inputs")

  def _slice(x):
    return x[:, :encoder_dim], x[:, encoder_dim:]

  def augment_unlabeled(u):
    u = tf.nn.dropout(u, rate=0.25)
    # u = add_gaussian_noise(u, gamma=0.1)
    u = batch_interpolation(u, alpha=0.9, random=True)
    return u

  labels = tf.placeholder(tf.float32, (None,), name="labels")
  logits = model.forward(*_slice(inputs))
  dot = tf.reduce_sum(tf.multiply(*_slice(inputs)), axis=1)
  loss_xe = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                    logits=logits)
  loss_xe = tf.reduce_mean(loss_xe)

  logits_us = []
  for _ in range(k):
    unlabeled_a, unlabeled_b = _slice(u_inputs)
    u_a = augment_unlabeled(unlabeled_a)  # augment by dropout
    u_b = augment_unlabeled(unlabeled_b)  # augment by dropout
    logits_u = model.forward(u_a, u_b)
    logits_us.append(logits_u)

  guess = guess_label(logits_us, temp=temp, binary=True)
  lu = tf.stop_gradient(guess)

  labels_us = tf.concat([lu] * k, axis=0)
  logits_us = tf.concat(logits_us, axis=0)

  loss_l2u = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_us,
                                                     logits=logits_us)
  loss_l2u = tf.reduce_mean(loss_l2u)

  loss = loss_xe + lambda_u * loss_l2u
  t_vars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars, global_step=tf.train.get_or_create_global_step())

  member_embeds, nonmember_embeds = load_text_embedding(
    train_filenames, test_filenames, freq_min)

  membership_labels = np.concatenate(
    [np.ones(len(member_embeds)), np.zeros(len(nonmember_embeds))])

  train_embeds, test_embeds, train_labels, test_labels = train_test_split(
    member_embeds + nonmember_embeds, membership_labels, random_state=12345,
    test_size=0.5, stratify=membership_labels)

  train_embeds, unlabeled_embeds, train_labels, _ = train_test_split(
    train_embeds, train_labels, random_state=12345,
    train_size=FLAGS.train_size / 0.5, stratify=train_labels)

  print('Labeled {} and labeled {} unlabeled books...'.format(
    len(train_embeds), len(unlabeled_embeds)))

  train_y = []
  for emb, label in zip(train_embeds, train_labels):
    train_y.append(np.ones(len(emb)) * label)

  train_y = np.concatenate(train_y).astype(np.float32)
  train_x = np.vstack(train_embeds)
  unlabeled_x = np.vstack(unlabeled_embeds)
  unlabeled_data_sampler = inf_batch_iterator(len(unlabeled_x), batch_size)
  print('Training attack model {} and labeled {} unlabeled...'.format(
    len(train_x), len(unlabeled_x)))

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    def calculate_adversarial_advantage(baseline):
      fetch = dot if baseline else logits
      test_member_metrics, test_nonmember_metrics = collect_scores(
        inputs, test_embeds, test_labels, sess, fetch)

      compute_adversarial_advantage(np.concatenate(test_member_metrics),
                                    np.concatenate(test_nonmember_metrics))
      if FLAGS.book_level:
        train_member_metrics, train_nonmember_metrics = collect_scores(
          inputs, train_embeds, train_labels, sess, fetch)
        all_metrics_flat = np.concatenate(
          train_member_metrics + train_nonmember_metrics
          + test_member_metrics + test_nonmember_metrics)
        metric_range = (np.min(all_metrics_flat), np.max(all_metrics_flat))

        data = (train_member_metrics, train_nonmember_metrics,
                test_member_metrics, test_nonmember_metrics)
        adversarial_advantage_from_trained(data, metric_range)

    calculate_adversarial_advantage(True)
    for epoch in range(50):
      iterations = 0
      train_loss_xe = 0
      train_loss_l2u = 0

      for batch_idx in iterate_minibatches_indices(
        len(train_y), batch_size=batch_size, shuffle=True):
        batch_u_idx = next(unlabeled_data_sampler)
        feed = {inputs: train_x[batch_idx],
                u_inputs: unlabeled_x[batch_u_idx],
                labels: train_y[batch_idx]}
        err_xe, err_l2u, _ = sess.run([loss_xe, loss_l2u, train_ops],
                                      feed_dict=feed)
        train_loss_xe += err_xe
        train_loss_l2u += err_l2u
        iterations += 1

      log("\nEpoch: {}, Loss xe: {:.4f}, Loss l2u: {:.4f}".format(
          epoch, train_loss_xe / iterations, train_loss_l2u / iterations))
      calculate_adversarial_advantage(False)


def main(unused_argv):
  train_semisupervised(temp=FLAGS.temp, lambda_u=FLAGS.lambda_u, k=FLAGS.k)


if __name__ == '__main__':
  app.run(main)
