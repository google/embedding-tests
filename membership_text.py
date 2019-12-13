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
from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

from data.bookcorpus import split_bookcorpus
from data.common import MODEL_DIR, EMB_DIR
from membership.models import BilinearMetricModel, LinearMetricModel, \
  DeepSetModel
from membership.utils import compute_adversarial_advantage
from thought import get_model_config, get_model_ckpt_name
from utils.common_utils import log
from utils.sent_utils import iterate_minibatches_indices, load_embeds, \
  load_raw_sents

tf.logging.set_verbosity(tf.logging.FATAL)

flags.DEFINE_integer('epoch', 0, 'Epochs of training')
flags.DEFINE_integer('freq_min', 90,
                     'use rare words above this percentile, e.g. 80=the most '
                     'infrequent 20 percent sentences')
flags.DEFINE_integer('k', 1, 'Number of augmentations')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('n_shard', 7400, 'Number of books to test')
flags.DEFINE_integer('train_size', 1500, 'Number of books for training the '
                                         'membership inference attack model')
flags.DEFINE_float('temp', 0.5, 'Temp for sharpening')
flags.DEFINE_float('lambda_u', 1.0, 'Lambda for unlabeled data loss')
flags.DEFINE_float('wd', 1e-4, 'Weight decay for membership model')
flags.DEFINE_string('metric', 'dot', 'Similarity metric')
flags.DEFINE_string('emb_dir',  EMB_DIR,
                    'Feature directory for saving embedding model')
flags.DEFINE_string('model_dir',  os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('model_name', 'quickthought', 'Model name')
flags.DEFINE_string('model', 'linear', 'Attack model')

flags.DEFINE_string('data_name', 'bookcorpus', 'Data name')
flags.DEFINE_boolean('book_level', False, 'Evaluate on book level')
flags.DEFINE_boolean('ssl', False, 'Use semisupervised learning')

FLAGS = flags.FLAGS


def merge_context(embs_a, embs_b):
  merged = []
  for a, b in zip(embs_a, embs_b):
    merged.append(np.hstack([a, b]))
  return merged


def load_freq_weights(filenames, freq_min):
  sents_dir = os.path.join(
    FLAGS.emb_dir, 'bookcorpus_raw_rare{}'.format(freq_min))

  sents, masks = load_raw_sents(
    filenames, sents_dir, rtn_filenames=False, stack=False)

  weights = []
  for ss, mm in zip(sents, masks):
    freqs = np.sum(ss, axis=1).astype(np.float)
    lengths = np.sum(mm, axis=1).astype(np.float)
    weights.append(freqs / lengths)
  return weights


def load_book_embedding(train_filenames, test_filenames, freq_min):
  model_name = get_model_ckpt_name(FLAGS.model_name, epoch=FLAGS.epoch)
  train_filenames = train_filenames[:FLAGS.n_shard]
  test_filenames = test_filenames[:FLAGS.n_shard]
  feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}'.format(
    model_name, freq_min))
  ctx_feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}_ctx'.format(
    model_name, freq_min))

  print('Loading embeddings from {}...'.format(feat_dir))

  member_embeds_a = load_embeds(train_filenames, feat_dir, stack=False)
  member_embeds_b = load_embeds(train_filenames, ctx_feat_dir, stack=False)

  assert len(member_embeds_a) == len(member_embeds_b)
  member_embeds = merge_context(member_embeds_a, member_embeds_b)
  del member_embeds_a, member_embeds_b

  nonmember_embeds_a = load_embeds(test_filenames, feat_dir, stack=False)
  nonmember_embeds_b = load_embeds(test_filenames, ctx_feat_dir, stack=False)
  assert len(nonmember_embeds_a) == len(nonmember_embeds_b)
  nonmember_embeds = merge_context(nonmember_embeds_a, nonmember_embeds_b)
  del nonmember_embeds_a, nonmember_embeds_b

  return member_embeds, nonmember_embeds


def collect_scores(inputs, embs, sess, fetch, training=None):
  is_stacked = not isinstance(embs, list)
  if not is_stacked:
    stacked_embs = np.vstack(embs)
  else:
    stacked_embs = embs

  stacked_scores = []
  encoder_dim = stacked_embs.shape[1] // 2
  for batch_idx in iterate_minibatches_indices(
    len(stacked_embs), batch_size=1024, shuffle=False):
    feed = {inputs[0]: stacked_embs[batch_idx][:, :encoder_dim],
            inputs[1]: stacked_embs[batch_idx][:, encoder_dim:]}
    if training is not None:
      feed[training] = False

    scores = sess.run(fetch, feed_dict=feed)
    stacked_scores.append(scores)

  stacked_scores = np.concatenate(stacked_scores)

  start_idx = 0
  metrics = []
  for emb in embs:
    if is_stacked:
      scores = stacked_scores[start_idx: start_idx + 1]
      start_idx += 1
    else:
      scores = stacked_scores[start_idx: start_idx + len(emb)]
      start_idx += len(emb)
    metrics.append(scores)

  return metrics


def membership_split(data):
  embs, labels = data
  indices = np.arange(len(embs))
  train_indices, test_indices = train_test_split(
    indices, random_state=12345, test_size=0.8, stratify=labels)

  train_indices, unlabeled_indices = train_test_split(
    train_indices, random_state=12345,
    train_size=FLAGS.train_size, stratify=labels[train_indices])
  return train_indices, test_indices, unlabeled_indices


def trained_metric_attack():
  freq_min = FLAGS.freq_min

  # load data part
  train_filenames, test_filenames = split_bookcorpus(0)
  member_embeds, nonmember_embeds = load_book_embedding(train_filenames,
                                                        test_filenames,
                                                        freq_min)

  membership_labels = np.concatenate(
    [np.ones(len(member_embeds)), np.zeros(len(nonmember_embeds))])

  all_embeds = member_embeds + nonmember_embeds
  train_indices, test_indices, _ = membership_split(
      (all_embeds, membership_labels))

  def indices_to_data(indices):
    embeds, labels, weights = [], [], []
    for idx in indices:
      embeds.append(all_embeds[idx])
      labels.append(membership_labels[idx])

    return embeds, labels

  train_embeds, train_labels = indices_to_data(train_indices)
  test_embeds, test_labels = indices_to_data(test_indices)

  train_y = []
  for emb, label in zip(train_embeds, train_labels):
    train_y.append(np.ones(len(emb)) * label)

  train_y = np.concatenate(train_y).astype(np.float32)
  train_x = np.vstack(train_embeds)

  # define attack model
  config = get_model_config(FLAGS.model_name)
  encoder_dim = config["encoder_dim"]

  optimizer = tf.train.AdamOptimizer(1e-4)

  inputs_a = tf.placeholder(tf.float32, (None, encoder_dim), name="inputs_a")
  inputs_b = tf.placeholder(tf.float32, (None, encoder_dim), name="inputs_b")
  labels = tf.placeholder(tf.float32, (None,), name="labels")
  training = tf.placeholder(tf.bool, name="training")

  if FLAGS.model == 'deepset':
    model = DeepSetModel(encoder_dim // 2)
  elif FLAGS.model == 'bilinear':
    model = BilinearMetricModel(encoder_dim)
  elif FLAGS.model == 'linear':
    model = LinearMetricModel(encoder_dim // 2)
  else:
    raise ValueError(FLAGS.model)

  logits = model.forward(inputs_a, inputs_b, training=training)
  learned_sim = logits

  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.reduce_mean(loss)

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

  t_vars = tf.trainable_variables()
  post_ops = [tf.assign(v, v * (1 - FLAGS.wd)) for v in t_vars if
              'kernel' in v.name]

  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars,  global_step=tf.train.get_or_create_global_step())

  with tf.control_dependencies([train_ops]):
    train_ops = tf.group(*post_ops)

  inputs = [inputs_a, inputs_b]
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    def split_metrics(ms, ls):
      member_ms, nonmember_ms = [], []
      for m, l in zip(ms, ls):
        if l == 1:
          member_ms.append(m)
        else:
          nonmember_ms.append(m)

      return member_ms, nonmember_ms

    def weighted_average(x):
      return np.mean(x)

    def calculate_adversarial_advantage(fetch):
      test_metrics = collect_scores(inputs, test_embeds, sess, fetch, training)
      test_member_ms, test_nonmember_ms = split_metrics(test_metrics,
                                                        test_labels)

      compute_adversarial_advantage(np.concatenate(test_member_ms),
                                    np.concatenate(test_nonmember_ms))

      if FLAGS.book_level:
        compute_adversarial_advantage(
          [weighted_average(m) for m in test_member_ms],
          [weighted_average(m) for m in test_nonmember_ms])

    calculate_adversarial_advantage(sim)
    print('Training attack model with {} embs...'.format(len(train_y)))
    for epoch in range(10):
      iterations = 0
      train_loss = 0

      for batch_idx in iterate_minibatches_indices(
        len(train_y), batch_size=FLAGS.batch_size, shuffle=True):
        feed = {inputs_a: train_x[batch_idx][:, :encoder_dim],
                inputs_b: train_x[batch_idx][:, encoder_dim:],
                labels: train_y[batch_idx], training: True}
        err, _ = sess.run([loss, train_ops], feed_dict=feed)
        train_loss += err
        iterations += 1

      log("\nEpoch: {}, Loss: {:.4f}".format(epoch, train_loss / iterations))
      calculate_adversarial_advantage(learned_sim)


def main(_):
  trained_metric_attack()


if __name__ == '__main__':
  app.run(main)
