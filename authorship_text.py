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
from collections import Counter

import tensorflow as tf
import tensorflow_hub as hub
import tqdm
from absl import app
from absl import flags

from attribute.mixmatch import guess_label, MixMode, interleave
from attribute.models import TextCNN, TextCharCNN, build_model, build_ae_model
from attribute.utils import acc_metrics, get_attrs_to_ids, \
  batch_interpolation, add_gaussian_noise
from data.bookcorpus import load_author_data as bookcorpus_author_data, \
  build_vocabulary as bookcorpus_vocab
from data.common import MODEL_DIR
from data.reddit import load_author_data as reddit_author_data
from thought import get_model_ckpt_name, get_model_config
from thought.quick_thought_model import QuickThoughtModel
from thought.vocabulary_expansion import expand_vocabulary
from utils.common_utils import log
from utils.sent_utils import iterate_minibatches_indices, inf_batch_iterator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags.DEFINE_integer('train_size', 50, 'Number of training data per attribute')
flags.DEFINE_integer('test_size', 250, 'Number of test data per attribute')
flags.DEFINE_integer('unlabeled_size', 0, 'Number of unlabeled data per '
                                          'attribute')
flags.DEFINE_integer('hidden_size', 0, 'Hidden size for attack model')
flags.DEFINE_integer('epochs', 100, 'Epochs of training')
flags.DEFINE_integer('encoder_epoch', 0, 'Epochs of quickthought')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('u_batch_size', 512, 'Batch size')
flags.DEFINE_integer('top_attr', 800, 'Predict top k attr')
flags.DEFINE_integer('k', 2, 'Number of augmentation in mixmatch')
flags.DEFINE_float('temp', 0.5, 'Temp for sharpening')
flags.DEFINE_float('lambda_u', 0.1, 'Lambda for unlabeled data loss')
flags.DEFINE_float('beta', 0.75, 'Beta for beta distribution in mixmatch')
flags.DEFINE_float('lr', 1e-3, 'Learning rate for attribution model')
flags.DEFINE_float('wd', 1e-4, 'Weight decay for attribution model')
flags.DEFINE_float('gamma', 0., 'Loss ratio for adversarial')
flags.DEFINE_string('algo', 'mixmatch', 'Semi-supervised algorithm')
flags.DEFINE_string('mixmode', '.', 'Mix mode for mixmatch')
flags.DEFINE_string('model_dir',  os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('model_name', 'quickthought', 'Model name')
flags.DEFINE_string('data_name', 'bookcorpus', 'Data name')
flags.DEFINE_boolean('norm', True, 'Normalize embedding')
flags.DEFINE_boolean('tune', False, 'Tune or freeze encoder when finetune')
flags.DEFINE_boolean('interleave', False, 'Whether to interleave batch in '
                                          'mixmatch')

FLAGS = flags.FLAGS


MAX_LEN = {
  'reddit': 64,
  'bookcorpus': 50
}


def build_vocabulary(train_sents, test_sents, unlabeled_sents=(),
                     min_word_count=2):
  all_train_words = [w for sent in train_sents for w in sent]
  all_test_words = [w for sent in test_sents for w in sent]
  all_words = all_train_words + all_test_words
  if len(unlabeled_sents):
    all_words += [w for sent in unlabeled_sents for w in sent]

  word_count = Counter(all_words)
  vocab = dict()
  idx = 1
  for word, count in word_count.most_common():
    if count >= min_word_count:
      vocab[word] = idx
      idx += 1
  return vocab


def preprocess_raw_data(train_sents, test_sents, unlabeled_sents=(),
                        vocab=None, min_word_count=2):
  if vocab is None:
    vocab = build_vocabulary(train_sents, test_sents, unlabeled_sents,
                             min_word_count)
  max_len = MAX_LEN[FLAGS.data_name]

  def sents_to_indices(sents):
    x = np.zeros((len(sents), max_len), dtype=np.int64)
    m = np.zeros((len(sents), max_len), dtype=np.int32)

    for i, sent in enumerate(sents):
      word_indices = [vocab.get(w, 0) for w in sent]
      length = min(len(word_indices), max_len)
      x[i, :length] = word_indices[:length]
      m[i, :length] = 1
    return x, m

  train_x, train_m = sents_to_indices(train_sents)
  test_x, test_m = sents_to_indices(test_sents)

  if len(unlabeled_sents):
    unlabeled_x, unlabeled_m = sents_to_indices(unlabeled_sents)
    return train_x, train_m, test_x, test_m, unlabeled_x, unlabeled_m

  return train_x, train_m, test_x, test_m


def train_loops(epochs, n_train, n_test, train_fn, eval_fn, batch_size,
                n_unlabeled=0, interleave_batch=False):
  if n_unlabeled:
    include_last = not interleave_batch
    u_batch_size = FLAGS.u_batch_size if include_last else batch_size
    unlabeled_data_sampler = inf_batch_iterator(n_unlabeled, u_batch_size)
  else:
    include_last = True
    unlabeled_data_sampler = None

  for epoch in range(epochs):
    train_iterations = 0
    train_loss = 0
    train_u_loss = 0

    for batch_idx in iterate_minibatches_indices(n_train, batch_size, True,
                                                 include_last=include_last):
      if unlabeled_data_sampler is None:
        err = train_fn(batch_idx)
      else:
        batch_u_idx = next(unlabeled_data_sampler)
        err, err_u = train_fn(batch_idx, batch_u_idx)
        train_u_loss += err_u

      train_loss += err
      train_iterations += 1

    test_loss = 0
    test_acc = 0
    test_top5_acc = 0
    test_iterations = 0
    for batch_idx in iterate_minibatches_indices(n_test, 512, False):
      err, acc, top5_acc = eval_fn(batch_idx)
      test_acc += acc
      test_top5_acc += top5_acc
      test_loss += err
      test_iterations += 1

    if (epoch + 1) % 10 == 0:
      log("Epoch: {}, train loss: {:.4f}, train l2u loss {:.4f}, "
          "test loss={:.4f}, test acc={:.2f}%, test top5 acc={:.2f}%".format(
            epoch + 1, train_loss / train_iterations,
            train_u_loss / train_iterations,
            test_loss / test_iterations,
            test_acc / n_test * 100,
            test_top5_acc / n_test * 100))


def train_text_cnn(data, num_attr):
  lr = FLAGS.lr
  batch_size = FLAGS.batch_size

  train_sents, train_y, test_sents, test_y = data
  train_x, train_m, test_x, test_m = preprocess_raw_data(
    train_sents, test_sents)

  inputs = tf.placeholder(tf.int64, (None, None), name="inputs")
  masks = tf.placeholder(tf.int32, (None, None), name="masks")

  labels = tf.placeholder(tf.int64, (None,), name="labels")
  training = tf.placeholder(tf.bool, name='training')

  text_cnn = TextCNN(vocab_size=50001, emb_dim=100, num_filter=128,
                     init_word_emb=None)
  classifier = build_model(num_attr, FLAGS.hidden_size)

  model_fn = lambda x, m, t: classifier(text_cnn.forward(x, m, t), t)

  logits = model_fn(inputs, masks, training)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=logits)
  loss = tf.reduce_mean(loss)
  opt_loss = loss
  accuracies, top5_accuracies, predictions = acc_metrics(logits, labels,
                                                         num_attr)
  eval_fetches = [loss, accuracies, top5_accuracies]

  t_vars = tf.trainable_variables()
  post_ops = [tf.assign(v, v * (1 - FLAGS.wd)) for v in t_vars if
              'kernel' in v.name]

  optimizer = tf.train.AdamOptimizer(lr)
  grads_and_vars = optimizer.compute_gradients(opt_loss, t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars, global_step=tf.train.get_or_create_global_step())

  with tf.control_dependencies([train_ops]):
    train_ops = tf.group(*post_ops)

  log('Train attack model...')
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    def train_fn(batch_idx):
      feed = {inputs: train_x[batch_idx], masks: train_m[batch_idx],
              labels: train_y[batch_idx], training: True}
      err, _ = sess.run([loss, train_ops], feed_dict=feed)
      return err

    def eval_fn(batch_idx):
      feed = {inputs: test_x[batch_idx], masks: test_m[batch_idx],
              labels: test_y[batch_idx], training: False}
      return sess.run(eval_fetches,  feed_dict=feed)

    n_train, n_test = len(train_y), len(test_y)
    train_loops(FLAGS.epochs, n_train, n_test, train_fn, eval_fn, batch_size)


def train_text_char_cnn(data, num_attr):
  lr = FLAGS.lr
  batch_size = FLAGS.batch_size

  train_sents, train_y, test_sents, test_y = data
  num_chars = 129
  max_char_len = 400

  def sents_to_chars(sents):
    max_len = 0
    chars = np.ones((len(sents), max_char_len), dtype=np.int64) * num_chars
    for i, sent in enumerate(sents):
      sent_chars = [ord(c) for c in str(sent)]
      max_len = max(len(sent_chars), max_len)
      if len(sent_chars) > max_char_len:
        sent_chars = sent_chars[:max_char_len]
      chars[i, :len(sent_chars)] = sent_chars
    return chars

  train_x = sents_to_chars(train_sents)
  test_x = sents_to_chars(test_sents)

  inputs = tf.placeholder(tf.int64, (None, max_char_len), name="inputs")
  labels = tf.placeholder(tf.int64, (None,), name="labels")
  training = tf.placeholder(tf.bool, name='training')

  text_cnn = TextCharCNN(num_chars, hidden_size=512, num_filter=128)
  classifier = build_model(num_attr, FLAGS.hidden_size)

  model_fn = lambda x, t: classifier(text_cnn.forward(x, t), t)

  logits = model_fn(inputs, training)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=logits)
  loss = tf.reduce_mean(loss)
  opt_loss = loss
  accuracies, top5_accuracies, predictions = acc_metrics(logits, labels,
                                                         num_attr)
  eval_fetches = [loss, accuracies, top5_accuracies]

  t_vars = tf.trainable_variables()
  post_ops = [tf.assign(v, v * (1 - FLAGS.wd)) for v in t_vars if
              'kernel' in v.name]

  optimizer = tf.train.AdamOptimizer(lr)
  grads_and_vars = optimizer.compute_gradients(opt_loss, t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars, global_step=tf.train.get_or_create_global_step())

  with tf.control_dependencies([train_ops]):
    train_ops = tf.group(*post_ops)

  log('Train attack model...')
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    def train_fn(batch_idx):
      feed = {inputs: train_x[batch_idx], labels: train_y[batch_idx],
              training: True}
      err, _ = sess.run([loss, train_ops], feed_dict=feed)
      return err

    def eval_fn(batch_idx):
      feed = {inputs: test_x[batch_idx], labels: test_y[batch_idx],
              training: False}
      return sess.run(eval_fetches,  feed_dict=feed)

    n_train, n_test = len(train_y), len(test_y)
    train_loops(FLAGS.epochs, n_train, n_test, train_fn, eval_fn, batch_size)


def encode_sentences(train_sents, test_sents, unlabeled_sents):
  query_size = 2048
  vocab = bookcorpus_vocab(0, rebuild=False)
  local_models = {'quickthought', 'transformer'}

  log('Encoding sentences...')
  if FLAGS.model_name in local_models:
    ckpt_name = get_model_ckpt_name(FLAGS.model_name,
                                    epoch=FLAGS.encoder_epoch, batch_size=800,
                                    gamma=FLAGS.gamma, attr='author')

    model_path = os.path.join(FLAGS.model_dir, ckpt_name)
    config = get_model_config(FLAGS.model_name)
    vocab, init_word_emb = expand_vocabulary(model_path, vocab)
    vocab_size = len(vocab) + 1

    model = QuickThoughtModel(vocab_size, config['emb_dim'],
                              config['encoder_dim'], 1,  init_word_emb=None,
                              cell_type=config['cell_type'], train=False)

    inputs = tf.placeholder(tf.int64, (None, None), name='inputs')
    masks = tf.placeholder(tf.int32, (None, None), name='masks')
    encode_emb = tf.nn.embedding_lookup(model.word_in_emb, inputs)
    encoded = model.encode(encode_emb, masks, model.in_cells, model.proj_in)
    if FLAGS.norm:
      encoded = tf.nn.l2_normalize(encoded, axis=-1)
    # model_vars = tf.trainable_variables()
    model_vars = {v.name[:-2]: v
                  for v in tf.trainable_variables()
                  if not v.name.startswith('emb')}

    saver = tf.train.Saver(model_vars)
    sess = tf.Session()
    emb_plhdr = tf.placeholder(tf.float32,
                               shape=(vocab_size, config['emb_dim']))
    sess.run(model.word_in_emb.assign(emb_plhdr),
             {emb_plhdr: init_word_emb})

    print('Loading weight from {}'.format(model_path))
    saver.restore(sess, os.path.join(model_path, 'model.ckpt'))
    encoder_fn = lambda s: sess.run(encoded, feed_dict={inputs: s[0],
                                                        masks: s[1]})
  elif FLAGS.model_name == 'skipthought':
    from models.skip_thoughts import encoder_manager
    from models.skip_thoughts import configuration
    model_dir = os.path.join(NFS_DIR, 'models/skip/')
    vocab_file = os.path.join(model_dir, 'vocab.txt')
    embedding_file = os.path.join('./skip_thoughts/', 'embeddings.npy')
    ckpt_path = os.path.join(model_dir, 'model.ckpt-500008')
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(bidirectional_encoder=True,
                                                  shuffle_input_data=False),
                       vocabulary_file=vocab_file,
                       embedding_matrix_file=embedding_file,
                       checkpoint_path=ckpt_path)
    encoder_fn = lambda s: encoder.encode(s, batch_size=query_size,
                                          use_norm=False)
    sess = encoder.sessions[0]
  elif FLAGS.model_name == 'use':
    embed_module = 'https://tfhub.dev/google/' \
                   'universal-sentence-encoder-large/3'
    embed = hub.Module(embed_module, trainable=False)
    inputs = tf.placeholder(tf.string, shape=(None,))
    encoded = embed(inputs)
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    encoder_fn = lambda s: sess.run(encoded, feed_dict={inputs: s})
  elif FLAGS.model_name == 'elmo':
    query_size = 512
    embed_module = 'https://tfhub.dev/google/elmo/2'
    embed = hub.Module(embed_module, trainable=False)
    inputs = tf.placeholder(tf.string, shape=(None,))
    encoded = embed(inputs, signature='default', as_dict=True)['default']
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    encoder_fn = lambda s: sess.run(encoded, feed_dict={inputs: s})
  elif FLAGS.model_name == 'infersent':
    from models.infersent.models import InferSent
    import torch
    model_version = 2
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0,
                    'version': model_version}
    encoder = InferSent(params_model)
    encoder.load_state_dict(torch.load("./infersent/infersent%s.pkl" %
                                       model_version))
    encoder.cuda()
    encoder.set_w2v_path('./infersent/crawl-300d-2M.vec')
    encoder.build_vocab_k_words(K=100000)
    encoder_fn = lambda s: encoder.encode(s, tokenize=False)
  else:
    raise ValueError(FLAGS.model)

  def encode_sents(s, n):
    embs = []
    pbar = tqdm.tqdm(total=n)
    tuple_inputs = isinstance(s, tuple)
    for batch_idx in iterate_minibatches_indices(n, query_size, False):
      if tuple_inputs:
        batch_embs = encoder_fn((s[0][batch_idx], s[1][batch_idx]))
      else:
        batch_embs = encoder_fn(s[batch_idx])
      embs.append(batch_embs)
      pbar.update(len(batch_embs))
    pbar.close()
    return np.vstack(embs)

  n_train, n_test, n_unlabeled = len(train_sents), len(test_sents),\
                                 len(unlabeled_sents)

  unlabeled_embs = []
  if FLAGS.model_name in local_models:
    rtn = preprocess_raw_data(train_sents, test_sents, unlabeled_sents,
                              vocab=vocab)
    train_embs = encode_sents((rtn[0], rtn[1]), n_train)
    test_embs = encode_sents((rtn[2], rtn[3]), n_test)
    if n_unlabeled:
      unlabeled_embs = encode_sents((rtn[4], rtn[5]), n_unlabeled)
  else:
    train_embs = encode_sents(train_sents, n_train)
    test_embs = encode_sents(test_sents, n_test)
    if n_unlabeled:
      unlabeled_embs = encode_sents(unlabeled_sents, n_unlabeled)

  tf.keras.backend.clear_session()
  log('Encoded train {}, test {}'.format(train_embs.shape, test_embs.shape))
  return train_embs, test_embs, unlabeled_embs


def train_embedding_classifier(data, unlabeled_data, num_attr):
  batch_size = FLAGS.batch_size
  interleave_batch = FLAGS.interleave

  train_sents, train_y, test_sents, test_y = data
  train_embs, test_embs, unlabeled_embs = encode_sentences(
    train_sents, test_sents, unlabeled_data)

  semi_supervised = len(unlabeled_embs) > 0
  n_train, n_test = len(train_y), len(test_y)

  encoder_dim = train_embs.shape[1]
  inputs = tf.placeholder(tf.float32, (None, encoder_dim), name='inputs')
  unlabeled_inputs = tf.placeholder(tf.float32, (None, encoder_dim),
                                    name="u_inputs")
  labels = tf.placeholder(tf.int64, (None,), name='labels')

  training = tf.placeholder(tf.bool, name='training')
  model_fn = build_model(num_attr, FLAGS.hidden_size)

  def augment_unlabeled(u):
    u = tf.nn.dropout(u, rate=0.25)
    u = add_gaussian_noise(u, gamma=0.1)
    u = batch_interpolation(u, alpha=0.9, random=True)
    return u

  if not semi_supervised:
    logits = model_fn(inputs, training)
    accuracies, top5_accuracies, _ = acc_metrics(logits, labels, num_attr)
    loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                             logits=logits)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_l2u = tf.constant(0.)
    loss = loss_xe
    eval_loss = loss_xe
  elif FLAGS.algo == 'mixmatch':
    augment = MixMode(FLAGS.mixmode)
    us = []
    logits_us = []
    for _ in range(FLAGS.k):
      u = augment_unlabeled(unlabeled_inputs)
      logits_u = model_fn(u, training)
      logits_us.append(logits_u)
      us.append(u)

    guess = guess_label(logits_us, temp=FLAGS.temp)
    lu = tf.stop_gradient(guess)
    lx = tf.one_hot(labels, num_attr)

    xu, labels_xu = augment([inputs] + us, [lx] + [lu] * FLAGS.k,
                            [FLAGS.beta, FLAGS.beta])
    labels_x, labels_us = labels_xu[0], tf.concat(labels_xu[1:], 0)

    if interleave_batch:
      xu = interleave(xu, batch_size)

    logits_x = model_fn(xu[0], training)
    logits_us = []
    for u in xu[1:]:
      logits_u = model_fn(u, training)
      logits_us.append(logits_u)

    logits_xu = [logits_x] + logits_us
    if interleave_batch:
      logits_xu = interleave(logits_xu, batch_size)

    logits_x = logits_xu[0]
    loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x,
                                                         logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)

    logits_us = tf.concat(logits_xu[1:], 0)
    loss_l2u = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_us,
                                                          logits=logits_us)
    # loss_l2u = tf.square(labels_us - tf.nn.softmax(logits_us))
    loss_l2u = tf.reduce_mean(loss_l2u)
    global_step = tf.train.get_or_create_global_step()
    w_match = tf.clip_by_value(
      tf.cast(global_step, tf.float32) /
      (FLAGS.epochs * (int(n_train // batch_size) + 1)), 0, 1)

    loss = FLAGS.lambda_u * w_match * loss_l2u + loss_xe
    test_logits = model_fn(inputs, training)
    test_loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=test_logits)
    accuracies, top5_accuracies, _ = acc_metrics(test_logits, labels, num_attr)
    eval_loss = tf.reduce_mean(test_loss_xe)
  elif FLAGS.algo == 'uda':
    model_fn = build_ae_model(num_attr, 256, encoder_dim)
    us = []
    logits_us = []
    for _ in range(FLAGS.k):
      u = augment_unlabeled(unlabeled_inputs)
      logits_u = model_fn(u, training)[0]
      logits_us.append(logits_u)
      us.append(u)

    labels_u = guess_label(logits_us, temp=FLAGS.temp)
    labels_us = tf.concat([labels_u] * FLAGS.k, 0)

    logits_x, recon_x = model_fn(inputs, training)

    logits_us = []
    recon_us = []
    for u in us:
      logits_u, recon_u = model_fn(u, training)
      logits_us.append(logits_u)
      recon_us.append(recon_u)

    recon_loss = tf.reduce_mean(
      tf.reduce_sum(tf.square(inputs - recon_x), axis=-1))
    recon_us = tf.concat(recon_us, 0)
    us = tf.concat(us, 0)
    recon_loss += tf.reduce_mean(
      tf.reduce_sum(tf.square(us - recon_us), axis=-1))

    loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                             logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)

    logits_us = tf.concat(logits_us, 0)
    loss_l2u = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_us,
                                                          logits=logits_us)
    # loss_l2u = tf.square(labels_us - tf.nn.softmax(logits_us))
    loss_l2u = tf.reduce_mean(loss_l2u)
    global_step = tf.train.get_or_create_global_step()
    w_match = tf.clip_by_value(
      tf.cast(global_step, tf.float32) /
      (FLAGS.epochs * (int(n_train // batch_size) + 1)), 0, 1)

    loss = FLAGS.lambda_u * w_match * loss_l2u + loss_xe + recon_loss
    test_logits = model_fn(inputs, training)[0]
    test_loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=test_logits)
    accuracies, top5_accuracies, _ = acc_metrics(test_logits, labels, num_attr)
    eval_loss = tf.reduce_mean(test_loss_xe)
  else:
    raise ValueError(FLAGS.algo)

  eval_fetches = [eval_loss, accuracies, top5_accuracies]
  t_vars = tf.trainable_variables()
  post_ops = [tf.assign(v, v * (1 - FLAGS.wd)) for v in t_vars if
              'kernel' in v.name]

  optimizer = tf.train.AdamOptimizer(FLAGS.lr)
  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
      grads_and_vars, global_step=tf.train.get_or_create_global_step())
  with tf.control_dependencies([train_ops]):
    train_ops = tf.group(*post_ops)

  log('Train attack model...')
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    def train_fn(*batch_idx):
      if len(batch_idx) == 1:
        batch_idx = batch_idx[0]
        feed = {inputs: train_embs[batch_idx], labels: train_y[batch_idx],
                training: True}
      else:
        feed = {inputs: train_embs[batch_idx[0]], labels: train_y[batch_idx[0]],
                unlabeled_inputs: unlabeled_embs[batch_idx[1]], training: True}
      err_xe, err_l2u, _ = sess.run([loss_xe, loss_l2u, train_ops],
                                    feed_dict=feed)
      if semi_supervised:
        return err_xe, err_l2u
      return err_xe

    def eval_fn(batch_idx):
      feed = {inputs: test_embs[batch_idx],
              labels: test_y[batch_idx], training: False}
      return sess.run(eval_fetches, feed_dict=feed)

    train_loops(FLAGS.epochs, n_train, n_test, train_fn, eval_fn, batch_size,
                len(unlabeled_embs), interleave_batch)


def main(_):
  split_word = FLAGS.model_name in {'textcnn', 'quickthought', 'transformer'}
  if FLAGS.data_name == 'bookcorpus':
    train_sents, train_authors, test_sents, test_authors,\
        unlabeled_sents, unlabeled_authors = bookcorpus_author_data(
          train_size=FLAGS.train_size, test_size=FLAGS.test_size,
          unlabeled_size=FLAGS.unlabeled_size, split_by_book=True,
          split_word=split_word, top_attr=FLAGS.top_attr, min_len=10)
  elif FLAGS.data_name == 'reddit':
    train_sents, train_authors, test_sents, test_authors,\
        unlabeled_sents, unlabeled_authors = reddit_author_data(
          train_size=FLAGS.train_size, test_size=FLAGS.test_size,
          unlabeled_size=FLAGS.unlabeled_size, split_word=split_word,
          top_attr=FLAGS.top_attr)
  else:
    raise ValueError(FLAGS.data_name)

  author_to_ids = get_attrs_to_ids(train_authors)

  train_y = np.asarray([author_to_ids[author] for author in train_authors],
                       dtype=np.int64)
  test_y = np.asarray([author_to_ids[author] for author in test_authors],
                      dtype=np.int64)
  num_attr = len(author_to_ids)

  log('{} training, {} testing'.format(len(train_y), len(test_y)))
  test_label_count = Counter(test_y)
  log('Majority baseline: {:.4f}% out of {} authors'.format(
      test_label_count.most_common(1)[0][1] / len(test_y) * 100,
      len(test_label_count)))

  data = train_sents, train_y, test_sents, test_y

  if FLAGS.model_name == 'textcnn':
    train_text_cnn(data, num_attr)
  elif FLAGS.model_name == 'charcnn':
    train_text_char_cnn(data, num_attr)
  else:
    train_embedding_classifier(data, unlabeled_sents, num_attr)


if __name__ == '__main__':
  app.run(main)
