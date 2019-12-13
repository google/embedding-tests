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

import os
import numpy as np
import tensorflow as tf
import cPickle as pkl

from sklearn.model_selection import train_test_split
from gensim.matutils import unitvec
from gensim.models import Word2Vec, FastText
from gensim.parsing.preprocessing import STOPWORDS
from joblib import Parallel, delayed
from scipy.special import expit

from data.common import MODEL_DIR
from data.wiki9 import split_wiki9_articles, WIKI9Articles
from utils.word_utils import load_tf_embedding, load_glove_model
from utils.sent_utils import iterate_minibatches_indices
from membership.utils import compute_adversarial_advantage
from membership.models import LinearMetricModel
from collections import Counter, defaultdict
from multiprocessing import Process, Pipe


flags.DEFINE_float('noise_multiplier', 0.,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 0., 'Clipping norm')
flags.DEFINE_float('train_size', 0.2, 'Ratio of data for training the '
                                      'membership inference attack model')
flags.DEFINE_integer('epoch', 4, 'Load model trained this epoch')
flags.DEFINE_integer('microbatches', 128, 'microbatches')
flags.DEFINE_integer('exp_id', 0, 'Experiment trial number')
flags.DEFINE_integer('n_jobs', 16, 'number of CPU cores for parallel '
                                   'collecting metrics')
flags.DEFINE_integer('window', 3, 'window size for a context of words')
flags.DEFINE_integer('freq_min', 80,
                     'use word frequency rank above this percentile, '
                     'e.g. 80=the most infrequent 20 percent words')
flags.DEFINE_integer('freq_max', 100, 'maximum frequency rank')
flags.DEFINE_string('metric', 'cosine', 'Metric to use for cosine similarity')
flags.DEFINE_string('model', 'w2v', 'Word embedding model')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 'w2v'),
                    'Model directory for embedding model')
flags.DEFINE_boolean('idf', False,
                     'Weight score by inverse document frequency')
flags.DEFINE_boolean('ctx_level', False,
                     'Context level or article level inference')
flags.DEFINE_boolean('learning', False,
                     'Whether to learning a metric model')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.FATAL)


def threshold_check(rank, thresh):
  if isinstance(thresh, tuple):
    return thresh[0] <= rank < thresh[1]
  else:
    assert isinstance(thresh, int)
    return thresh <= rank


def get_all_contexts(docs, model, thresh, window=3):
  vocab = model.wv.vocab

  all_words = sorted(vocab.keys())

  all_counts = 0.
  for word in all_words:
    all_counts += vocab[word].count

  all_docs_ctx = []
  for text in WIKI9Articles(docs, verbose=1):
    doc_ctx = []
    for i, word in enumerate(text):
      if word in vocab and threshold_check(vocab[word].index, thresh):
        s = max(0, i - window)
        e = min(len(text), i + window + 1)

        context = [neighbor for neighbor in text[s: i] + text[i + 1: e]
                   if neighbor not in STOPWORDS
                   and neighbor in vocab
                   and neighbor != word]

        if len(context) == 0:
          continue

        context_pair = [(vocab[word].index, vocab[neighbor].index)
                        for neighbor in context]
        doc_ctx.append(np.asarray(context_pair, dtype=np.int64))
        # all_docs_ctx.append(np.asarray(context_pair, dtype=np.int64))

    if len(doc_ctx) > 0:
      all_docs_ctx.append(doc_ctx)

  return all_docs_ctx


def split_docs(docs, n_jobs):
  n_docs = len(docs)
  n_docs_per_job = n_docs // n_jobs + 1
  splits = []
  for i in range(n_jobs):
    splits.append(docs[i * n_docs_per_job: (i + 1) * n_docs_per_job])
  return splits


def trained_metric(exp_id=0, n_jobs=1, freqs=(80, 100), window=3,
                   emb_model='ft'):
  train_docs, test_docs = split_wiki9_articles(exp_id)
  save_dir = FLAGS.save_dir

  model_name = 'wiki9_{}_{}.model'.format(emb_model, FLAGS.exp_id)
  model_path = os.path.join(save_dir, model_name)

  if emb_model == 'ft':
    model = FastText.load(model_path)
  elif emb_model == 'w2v':
    model = Word2Vec.load(model_path)
  elif emb_model == 'glove':
    model = load_glove_model(model_path)
  elif emb_model == 'tfw2v':
    model = load_tf_embedding(FLAGS.exp_id, save_dir=save_dir,
                              epoch=FLAGS.epoch,
                              noise_multiplier=FLAGS.noise_multiplier,
                              l2_norm_clip=FLAGS.l2_norm_clip,
                              microbatches=FLAGS.microbatches)
  else:
    raise ValueError('No such embedding model: {}'.format(emb_model))

  word_vectors = model.wv.vectors
  word_emb = tf.convert_to_tensor(word_vectors)
  metric_model = LinearMetricModel(word_vectors.shape[1])

  optimizer = tf.train.AdamOptimizer(5e-4)
  inputs_a = tf.placeholder(tf.int64, (None,), name="inputs_a")
  inputs_b = tf.placeholder(tf.int64, (None,), name="inputs_b")
  labels = tf.placeholder(tf.float32, (None,), name="labels")

  embs_a = tf.nn.embedding_lookup(word_emb, inputs_a)
  embs_b = tf.nn.embedding_lookup(word_emb, inputs_b)

  logits = metric_model.forward(embs_a, embs_b)

  if FLAGS.metric == 'cosine':
    embs_a = tf.nn.l2_normalize(embs_a, axis=1)
    embs_b = tf.nn.l2_normalize(embs_b, axis=1)

  dot = tf.reduce_sum(tf.multiply(embs_a, embs_b), axis=1)

  # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.keras.losses.hinge(labels, logits)
  loss = tf.reduce_mean(loss)

  t_vars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars,  global_step=tf.train.get_or_create_global_step())

  vocab_size = len(model.wv.vocab)
  thresh = (int(vocab_size * freqs[0] / 100), int(vocab_size * freqs[1] / 100))

  print("Loading contexts for membership inference")

  if n_jobs > 1:
    member_job_ctxs = Parallel(n_jobs)(delayed(get_all_contexts)(
      ds, model, thresh, window) for ds in split_docs(train_docs, n_jobs))
    nonmember_job_ctxs = Parallel(n_jobs)(delayed(get_all_contexts)(
      ds, model, thresh, window) for ds in split_docs(test_docs, n_jobs))
    member_ctxs = [ctxs for job_ctxs in member_job_ctxs
                   for ctxs in job_ctxs]
    nonmember_ctxs = [ctxs for job_ctxs in nonmember_job_ctxs
                      for ctxs in job_ctxs]
  else:
    member_ctxs = get_all_contexts(train_docs, model, thresh, window)
    nonmember_ctxs = get_all_contexts(test_docs, model, thresh, window)

  print("Loaded {} member and {} nonmember".format(
      len(member_ctxs), len(nonmember_ctxs)))

  membership_labels = np.concatenate(
    [np.ones(len(member_ctxs)), np.zeros(len(nonmember_ctxs))])

  train_ctxs, test_ctxs, train_labels, test_labels = train_test_split(
    member_ctxs + nonmember_ctxs, membership_labels, random_state=12345,
    train_size=FLAGS.train_size, stratify=membership_labels)

  def flatten_ctxs(ctxs, labels):
    flat_ctxs, flat_labels = [], []
    for doc_ctx, doc_label in zip(ctxs, labels):
      flat_ctxs += doc_ctx
      flat_labels.append(np.ones(len(doc_ctx)) * doc_label)
    return flat_ctxs, np.concatenate(flat_labels)

  train_ctxs, train_labels = flatten_ctxs(train_ctxs, train_labels)
  test_ctxs, test_labels = flatten_ctxs(test_ctxs, test_labels)

  train_y = []
  for ctxs, label in zip(train_ctxs, train_labels):
    train_y.append(np.ones(len(ctxs)) * label)

  train_y = np.concatenate(train_y).astype(np.float32)
  train_x = np.vstack(train_ctxs)

  def collect_scores(ctxs, labels, sess, baseline=False):
    stacked_ctxs = np.vstack(ctxs)
    stacked_scores = []
    for batch_idx in iterate_minibatches_indices(
        len(stacked_ctxs), batch_size=1024, shuffle=False):
      feed = {inputs_a: stacked_ctxs[batch_idx][:, 0],
              inputs_b: stacked_ctxs[batch_idx][:, 1]}
      scores = sess.run(dot if baseline else logits, feed_dict=feed)
      stacked_scores.append(scores)
    stacked_scores = np.concatenate(stacked_scores)

    member_metrics, nonmember_metrics = [], []
    start_idx = 0
    for ctx, label in zip(ctxs, labels):
      scores = stacked_scores[start_idx: start_idx + len(ctx)]
      start_idx += len(ctx)

      if label == 1:
        member_metrics.append(scores)
      else:
        nonmember_metrics.append(scores)
    return member_metrics, nonmember_metrics

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    test_member_metrics, test_nonmember_metrics = collect_scores(
      test_ctxs, test_labels, sess, True)

    compute_adversarial_advantage(
        [np.mean(m) for m in test_member_metrics],
        [np.mean(m) for m in test_nonmember_metrics])

    print('Training attack model with {} data...'.format(len(train_x)))
    for epoch in range(30):
      iterations = 0
      train_loss = 0

      for batch_idx in iterate_minibatches_indices(
            len(train_y), batch_size=512, shuffle=True):
        feed = {inputs_a: train_x[batch_idx][:, 0],
                inputs_b: train_x[batch_idx][:, 1],
                labels: train_y[batch_idx]}
        err, _ = sess.run([loss, train_ops], feed_dict=feed)
        train_loss += err
        iterations += 1

      print("Epoch: {}, Loss: {:.4f}".format(epoch, train_loss / iterations))
      test_member_metrics, test_nonmember_metrics = collect_scores(
        test_ctxs, test_labels, sess)
      compute_adversarial_advantage(
          [np.mean(m) for m in test_member_metrics],
          [np.mean(m) for m in test_nonmember_metrics])


def metrics_rare_words(docs, model, thresh, window=3, metric='cosine',
                       ctx_level=True, idf_dict=None, pipe=None):
  vocab = model.wv.vocab

  all_words = sorted(vocab.keys())

  all_counts = 0.
  for word in all_words:
    all_counts += vocab[word].count

  metrics = []
  all_docs_metrics = []

  for text in WIKI9Articles(docs, verbose=1):
    doc_metrics = []
    for i, word in enumerate(text):
      if word in vocab and threshold_check(vocab[word].index, thresh):
        s = max(0, i - window)
        e = min(len(text), i + window + 1)

        context = [neighbor for neighbor in text[s: i] + text[i + 1: e]
                   if neighbor not in STOPWORDS
                   and neighbor in vocab
                   and neighbor != word]

        if len(context) == 0:
          continue

        if metric == 'cosine':
          scores = [model.wv.similarity(word, neighbor)
                    for neighbor in context]
        elif metric == 'l2':
          context_embs = [unitvec(model.wv[neighbor])
                          for neighbor in context]
          emb = unitvec(model.wv[word])
          scores = [-np.linalg.norm(emb - context_emb)
                    for context_emb in context_embs]
        elif metric == 'dot':
          scores = [np.dot(model.wv[neighbor], model.wv[word])
                    for neighbor in context]
        elif metric == 'loglike':
          scores = [np.log(expit(np.dot(model.wv[neighbor], model.wv[word])))
                    for neighbor in context]
        else:
          raise ValueError('No such metric: {}'.format(metric))

        if idf_dict is not None:
          idf_weights = [idf_dict[neighbor] for neighbor in context]
          sum_weight = sum(idf_weights) / len(idf_weights)
          scores = [s * w / sum_weight for s, w in zip(scores, idf_weights)]

        doc_metrics += scores
        metrics.append(scores)

    if len(doc_metrics) > 0:
      all_docs_metrics.append(doc_metrics)

  result = np.asarray(metrics) if ctx_level else all_docs_metrics
  if pipe:
    pipe.send(result)
  else:
    return result


def get_idf_count(docs):
  idf_count = Counter()
  for text in WIKI9Articles(docs, verbose=1):
    unique_words = set(text)
    idf_count.update(unique_words)
  return idf_count


def get_idf_dict(docs, n_jobs=16):
  num_docs = len(docs)
  save_path = './data/wiki9_idf_count.pkl'

  if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
      idf_count = pkl.load(f)
  else:
    idf_counts = Parallel(n_jobs)(delayed(get_idf_count)(ds)
                                  for ds in split_docs(docs, n_jobs))
    idf_count = Counter()
    for count in idf_counts:
      idf_count.update(count)

    with open(save_path, 'wb') as f:
      pkl.dump(idf_count, f, -1)

  print('Compute idf dict...')
  idf_dict = defaultdict(lambda: np.log((num_docs + 1) / 1))
  idf_dict.update({idx: np.log((num_docs + 1) / (c + 1)) for (idx, c) in
                   idf_count.items()})
  return idf_dict


def run_normal_parallel(data, fn, n_jobs, *args):
  workers = []
  pipes = []
  for i in range(n_jobs):
    parent, child = Pipe()
    worker = Process(target=fn, args=(data[i], ) + args + (child, ))
    pipes.append(parent)
    worker.start()

  result = []
  for pipe in pipes:
    result.append(pipe.recv())

  for worker in workers:
    worker.terminate()

  return result


def find_signal(exp_id=0, n_jobs=1, freqs=(80, 100), window=3, metric='cosine',
                ctx_level=True, emb_model='ft'):
  train_docs, test_docs = split_wiki9_articles(exp_id)
  idf_dict = get_idf_dict(train_docs + test_docs) if FLAGS.idf else None

  save_dir = FLAGS.save_dir

  model_name = 'wiki9_{}_{}.model'.format(emb_model, FLAGS.exp_id)
  model_path = os.path.join(save_dir, model_name)

  if emb_model == 'ft':
    model = FastText.load(model_path)
  elif emb_model == 'w2v':
    model = Word2Vec.load(model_path)
  elif emb_model == 'glove':
    model = load_glove_model(model_path)
  elif emb_model == 'tfw2v':
    model = load_tf_embedding(FLAGS.exp_id, save_dir=save_dir,
                              epoch=FLAGS.epoch,
                              noise_multiplier=FLAGS.noise_multiplier,
                              l2_norm_clip=FLAGS.l2_norm_clip,
                              microbatches=FLAGS.microbatches)
  else:
    raise ValueError('No such embedding model: {}'.format(emb_model))

  vocab_size = len(model.wv.vocab)
  thresh = (int(vocab_size * freqs[0] / 100), int(vocab_size * freqs[1] / 100))
  if n_jobs > 1:
    args = model, thresh, window, metric, ctx_level, idf_dict
    train_metrics = run_normal_parallel(
      split_docs(train_docs, n_jobs),  metrics_rare_words, n_jobs, *args)
    train_metrics = np.concatenate(train_metrics)
    test_metrics = run_normal_parallel(
      split_docs(test_docs, n_jobs), metrics_rare_words, n_jobs, *args)
    test_metrics = np.concatenate(test_metrics)
  else:
    train_metrics = metrics_rare_words(train_docs, model, thresh, window,
                                       metric, ctx_level, idf_dict)
    test_metrics = metrics_rare_words(test_docs, model, thresh, window,
                                      metric, ctx_level, idf_dict)

  compute_adversarial_advantage([np.mean(m) for m in train_metrics],
                                [np.mean(m) for m in test_metrics])

  # if ctx_level:
  #   data = (train_metrics, test_metrics)
  #   adversarial_advantage_from_trained(data, histogram=False)
  # else:
  #   range_map = {
  #     'cosine': (-1, 1),
  #   }
  #   if metric in range_map:
  #     metric_range = range_map[metric]
  #   else:
  #     all_metrics = np.concatenate([np.concatenate(train_metrics),
  #                                   np.concatenate(test_metrics)])
  #     metric_range = (np.min(all_metrics), np.max(all_metrics))
  #
  #   data = (train_metrics, test_metrics)
  #   adversarial_advantage_from_trained(data, metric_range)


def main(unused_argv):
  if FLAGS.learning:
    trained_metric(FLAGS.exp_id, FLAGS.n_jobs,
                   (FLAGS.freq_min, FLAGS.freq_max),
                   FLAGS.window, FLAGS.model)
  else:
    find_signal(FLAGS.exp_id, FLAGS.n_jobs, (FLAGS.freq_min, FLAGS.freq_max),
                FLAGS.window, FLAGS.metric, FLAGS.ctx_level, FLAGS.model)


if __name__ == '__main__':
  app.run(main)
