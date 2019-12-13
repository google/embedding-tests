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

from absl import flags
from absl import app

import tensorflow as tf
import numpy as np
import time
import smart_open
import os

from dp_optimizer.dp_optimizer_sparse import SparseDPAdamGaussianOptimizer
from privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
from utils.common_utils import rigid_op_sequence
from data.wiki9 import read_wiki9_train_split
from data.common import MODEL_DIR
from utils.common_utils import log
from collections import Counter

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.5,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 0.25, 'Clipping norm')
flags.DEFINE_integer('exp_id', 0, 'Experiment trial number')
flags.DEFINE_integer('num_gpu', 1, 'Number of GPU')
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_integer('microbatches', 128, 'Number of microbatch')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')
flags.DEFINE_integer('hidden_size', 100, 'Number of hidden units')
flags.DEFINE_integer('n_sampled', 25, 'Number of hidden units')
flags.DEFINE_integer('print_every', 1000, 'Number of hidden units')
flags.DEFINE_boolean('dpsgd', True,
                     'If True, train with DP-SGD. '
                     'If False, train with vanilla SGD.')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 'w2v'),
                    'Model directory for embedding model')

FLAGS = flags.FLAGS


def compute_epsilon(steps, n):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / n
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps, orders=orders)
  # Delta is set to approximate 1 / (number of training points).
  return get_privacy_spent(orders, rdp, target_delta=1 / n)[0]


def create_lookup_tables(words):
  word_counts = Counter(words)
  sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
  int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

  return vocab_to_int, int_to_vocab


def get_target(words, idx, r):
  start = idx - r if (idx - r) > 0 else 0
  stop = idx + r
  target_words = words[start:idx] + words[idx + 1:stop + 1]
  return target_words


def get_batches(words, batch_size, word_sample_int, window_size=5):
  reduced_windows = np.random.randint(1, window_size + 1, size=len(words))
  sampled_int = np.random.rand(len(words)) * 2 ** 32

  for idx in range(0, len(words), batch_size):
    x, y = [], []
    batch = words[idx:idx + batch_size]
    batch_sampled_int = sampled_int[idx: idx + batch_size]
    rs = reduced_windows[idx: idx + batch_size]
    for ii in range(len(batch)):
      batch_x = batch[ii]
      if batch_sampled_int[ii] >= word_sample_int[batch_x]:
        continue
      batch_y = get_target(batch, ii, rs[ii])
      y.extend(batch_y)
      x.extend([batch_x] * len(batch_y))
    yield x, y


def preprocess_texts(exp_id=0, sample=1e-4):
  with smart_open.open(read_wiki9_train_split(exp_id), encoding='utf-8') as f:
    words = f.read().split()

  # Remove all words with  5 or fewer occurences
  word_counts = Counter(words)
  trimmed_words = [word for word in words if word_counts[word] >= 5]
  vocab_to_int, int_to_vocab = create_lookup_tables(trimmed_words)

  unigrams = []
  for i in range(len(int_to_vocab)):
    unigrams.append(word_counts[int_to_vocab[i]])

  retain_total = sum([word_counts[word] for word in vocab_to_int])
  word_sample_int = {}
  for word in vocab_to_int:
    v = word_counts[word]
    threshold_count = sample * retain_total
    word_probability = (np.sqrt(v / threshold_count) + 1) \
        * (threshold_count / v)
    word_probability = min(word_probability, 1.0)
    word_sample_int[vocab_to_int[word]] = int(round(word_probability * 2**32))

  train_words = [vocab_to_int[word] for word in trimmed_words]
  return train_words, unigrams, word_sample_int


def make_parallel(fn, optimizer, num_gpus, **kwargs):
  in_splits = {}
  for k, v in kwargs.items():
    in_splits[k] = tf.split(v, num_gpus)

  tower_grads = []
  tower_losses = []
  for i in range(num_gpus):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        out_grads, loss = fn(**{k: v[i] for k, v in in_splits.items()})
        tower_grads.append((lambda: optimizer.apply_gradients(out_grads)))
        tower_losses.append(loss)

  return tower_grads, tf.reduce_mean(tower_losses)


def main(_):
  assert FLAGS.dpsgd
  exp_id = FLAGS.exp_id
  num_gpu = FLAGS.num_gpu
  train_words, unigrams, word_sample_int = preprocess_texts(exp_id)
  n_vocab = len(unigrams)
  n_sampled = FLAGS.n_sampled
  n_embedding = FLAGS.hidden_size
  init_width = 0.5 / n_embedding
  epochs = FLAGS.epochs
  window_size = 5
  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate
  delta = 1 / len(train_words)

  cumtable = tf.constant(np.cumsum(unigrams))
  inputs = tf.placeholder(tf.int64, [None], name='inputs')
  labels = tf.placeholder(tf.int64, [None, 1], name='labels')
  embedding = tf.Variable(tf.random_uniform(
    (n_vocab, n_embedding), -init_width, init_width), name="emb")
  sm_w_t = embedding
  sm_b = tf.Variable(tf.zeros(n_vocab), name="sm_b")

  curr_words = tf.Variable(0, trainable=False)
  update_curr_words = curr_words.assign_add(batch_size)
  lr = learning_rate * tf.maximum(
    0.0001, 1.0 - tf.cast(curr_words, tf.float32) / len(train_words) / epochs)
  num_microbatches = FLAGS.microbatches

  if FLAGS.dpsgd:
    optimizer = SparseDPAdamGaussianOptimizer(
      l2_norm_clip=FLAGS.l2_norm_clip,
      noise_multiplier=FLAGS.noise_multiplier,
      num_microbatches=num_microbatches if num_microbatches > 0 else None,
      learning_rate=lr)
  else:
    optimizer = tf.train.AdamOptimizer(lr)

  t_vars = tf.trainable_variables()

  def model(x, y):
    nb = tf.shape(x)[0]
    example_emb = tf.nn.embedding_lookup(embedding, x)

    # Negative sampling.
    random_ints = tf.random.uniform((n_sampled * nb,),
                                    maxval=cumtable[-1], dtype=tf.int64)
    sampled_ids = tf.searchsorted(cumtable, random_ints, out_type=tf.int64)

    y_vec = tf.squeeze(y)
    true_w = tf.nn.embedding_lookup(sm_w_t, y_vec)
    true_b = tf.nn.embedding_lookup(sm_b, y_vec)
    true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

    sampled_w_mat = tf.reshape(sampled_w, [nb, n_sampled, n_embedding])
    sampled_b_vec = tf.reshape(sampled_b, [nb, n_sampled])
    example_emb_mat = tf.reshape(example_emb, [nb, n_embedding, 1])

    sampled_logits = tf.squeeze(
      tf.matmul(sampled_w_mat, example_emb_mat)) + sampled_b_vec

    # Calculate the loss using negative sampling
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    sampled_mask = 1 - tf.cast(
      tf.equal(y, tf.reshape(sampled_ids, [nb, n_sampled])), tf.float32)
    vector_loss = true_xent + tf.reduce_sum(sampled_xent * sampled_mask, axis=1)
    scalar_loss = tf.reduce_mean(vector_loss)

    if FLAGS.dpsgd:
      grads = optimizer.compute_gradients(
        vector_loss, t_vars, colocate_gradients_with_ops=num_gpu > 1)
    else:
      grads = optimizer.compute_gradients(
        scalar_loss, t_vars, colocate_gradients_with_ops=num_gpu > 1)

    return grads, scalar_loss

  if num_gpu > 1:
    tower_grads, scalar_loss = make_parallel(model, optimizer, num_gpu,
                                             x=inputs, y=labels)
    train_ops = rigid_op_sequence(tower_grads)
  else:
    grads_and_vars, scalar_loss = model(inputs, labels)
    train_ops = optimizer.apply_gradients(grads_and_vars)

  saver = tf.train.Saver()
  iterations = epochs * len(train_words) // batch_size
  print_every = FLAGS.print_every

  with tf.Session() as sess:
    iteration = 0
    train_loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
      start = time.time()
      for x, y in get_batches(train_words, batch_size, word_sample_int,
                              window_size):
        b = len(x)
        if num_microbatches > 0:
          offset = b - b % num_microbatches
          x, y = x[:offset], y[:offset]

        feed = {inputs: x, labels: np.array(y)[:, None]}
        err, _, _ = sess.run([scalar_loss, train_ops, update_curr_words],
                             feed_dict=feed)

        train_loss += err
        iteration += 1

        if iteration % print_every == 0:
          end = time.time()
          log("Iteration: {:.4f}%, Loss: {:.4f}, {:.4f} sec/batch".format(
            iteration / iterations * 100, train_loss / print_every,
            (end - start) / print_every))
          train_loss = 0
          start = time.time()
          if FLAGS.dpsgd:
            eps = compute_epsilon(iteration, len(train_words))
            log('The current epsilon is: {:.2f} for delta={}'.format(
                 eps, delta))

      model_name = 'tfw2v_{}'.format(exp_id)

      if FLAGS.dpsgd:
        model_name += 'e{}_n{}_l{}_mb{}'.format(
          e, FLAGS.noise_multiplier, FLAGS.l2_norm_clip, num_microbatches)
      eps = compute_epsilon(iteration, len(train_words))

      save_path = os.path.join(FLAGS.save_dir, model_name)
      if not os.path.exists(save_path):
        os.makedirs(save_path)

      saver.save(sess, os.path.join(save_path, "model.ckpt"))

      if FLAGS.dpsgd:
        with open(os.path.join(save_path, 'eps{:.2f}'.format(eps)), 'w'):
          pass


if __name__ == '__main__':
  app.run(main)
