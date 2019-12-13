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
from thought.quick_thought_model import QuickThoughtModel
from attribute.models import build_model, flip_gradient
import tensorflow as tf
import numpy as np
import time
import os
from invert.utils import tp_fp_fn_metrics
from utils.common_utils import log
from data.common import MODEL_DIR
from data.bookcorpus import load_bookcorpus_sentences, load_initialized_word_emb

tf.logging.set_verbosity(tf.logging.FATAL)

flags.DEFINE_integer('emb_dim', 620, 'embedding dimension')
flags.DEFINE_integer('encoder_dim', 1200, 'encoder dim')
flags.DEFINE_integer('context_size', 1, 'Context size')
flags.DEFINE_integer('batch_size', 800, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Epochs of training')
flags.DEFINE_integer('num_layer', 3, 'Number of transformer layer')
flags.DEFINE_integer('warmup_steps', 5000, 'Steps of warm up for transformers')
flags.DEFINE_integer('print_every', 100, 'Print loss every N iterations')
flags.DEFINE_float('lr', 0.0005, 'Learning rate for training')
flags.DEFINE_float('gamma', 0.3, 'Loss ratio for adversarial')
flags.DEFINE_string('attr', 'author', 'Attributes to censor')
flags.DEFINE_string('cell_type', 'LSTM', 'Encoder model')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')

FLAGS = flags.FLAGS


def iterate_triplet_minibatches(inputs, masks, labels, batch_size,
                                shuffle=True):
  input_indices = np.arange(len(inputs))[1:-1]
  parent_indices = input_indices - 1
  response_indices = input_indices + 1

  indices = np.arange(len(input_indices))
  if shuffle:
    np.random.shuffle(indices)

  num_batches = len(indices) // batch_size
  for batch_idx in range(num_batches):
    batch_indices = indices[batch_idx * batch_size:
                            (batch_idx + 1) * batch_size]
    batch_inputs = inputs[input_indices[batch_indices]],\
                   masks[input_indices[batch_indices]]
    batch_parents = inputs[parent_indices[batch_indices]],\
                    masks[parent_indices[batch_indices]]
    batch_responses = inputs[response_indices[batch_indices]],\
                      masks[response_indices[batch_indices]]
    batch_labels = labels[input_indices[batch_indices]]
    yield batch_inputs, batch_parents, batch_responses, batch_labels


def get_lr(global_step):
  # from attention is all you need
  global_step = tf.cast(global_step, tf.float32)
  arg1 = tf.math.rsqrt(global_step)
  arg2 = global_step * (FLAGS.warmup_steps ** -1.5)
  return tf.math.rsqrt(tf.cast(FLAGS.encoder_dim, tf.float32)) * \
      tf.math.minimum(arg1, arg2)


def main(_):
  epochs = FLAGS.epochs
  gamma = FLAGS.gamma
  batch_size = FLAGS.batch_size

  sents, sent_masks, authors, vocab = \
      load_bookcorpus_sentences(load_author=True)
  num_author = len(np.unique(authors))

  init_word_emb = load_initialized_word_emb()

  vocab_size = len(vocab) + 1

  log("training with {} sents and {} vocabs and {} authors".format(
    sents.shape, vocab_size, num_author))

  if init_word_emb is not None and init_word_emb.shape[1] != FLAGS.emb_dim:
    offset = FLAGS.emb_dim - init_word_emb.shape[1]
    if offset > 0:
      random_emb = np.random.uniform(-0.1, 0.1,
                                     (vocab_size, offset)).astype(np.float32)
      init_word_emb = np.hstack([init_word_emb, random_emb])
    else:
      init_word_emb = init_word_emb[:, :FLAGS.emb_dim]

  model = QuickThoughtModel(vocab_size, FLAGS.emb_dim,
                            FLAGS.encoder_dim, FLAGS.context_size,
                            cell_type=FLAGS.cell_type,
                            num_layer=FLAGS.num_layer,
                            init_word_emb=init_word_emb,
                            train=True, drop_p=0.15)

  global_step = tf.train.get_or_create_global_step()
  lr = get_lr(global_step) if FLAGS.cell_type == 'TRANS' else FLAGS.lr
  optimizer = tf.train.AdamOptimizer(lr)

  i_inputs = tf.placeholder(tf.int64, (None, None), name="i_inputs")
  i_masks = tf.placeholder(tf.int32, (None, None), name="i_masks")
  p_inputs = tf.placeholder(tf.int64, (None, None), name="p_inputs")
  p_masks = tf.placeholder(tf.int32, (None, None), name="p_masks")
  r_inputs = tf.placeholder(tf.int64, (None, None), name="r_inputs")
  r_masks = tf.placeholder(tf.int32, (None, None), name="r_masks")

  output_tensors = [(p_inputs, p_masks), (r_inputs, r_masks)]
  accs, loss = model.forward_triplet((i_inputs, i_masks),
                                     output_tensors, batch_size)

  thought_vector = model.thought_vector
  thought_vector = flip_gradient(thought_vector, gamma)

  if FLAGS.attr == 'author':
    labels = tf.placeholder(tf.int64, (None, ), name="labels")
    adv_model = build_model(num_author, FLAGS.encoder_dim // 2)
    adv_logits = adv_model(thought_vector, tf.constant(True, dtype=tf.bool))
    adv_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=adv_logits)
    adv_loss = tf.reduce_mean(adv_loss)
    adv_acc = tf.reduce_mean(tf.cast(
      tf.equal(labels, tf.argmax(adv_logits, axis=-1)), tf.float32))
  elif FLAGS.attr == 'word':
    labels = tf.placeholder(tf.float32, (None, None), name='labels')
    adv_model = build_model(vocab_size, FLAGS.encoder_dim // 2)
    adv_logits = adv_model(thought_vector, tf.constant(True, dtype=tf.bool))
    adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[:, 1:],
                                                       logits=adv_logits[:, 1:])
    adv_loss = tf.reduce_mean(tf.reduce_sum(adv_loss, axis=-1))
    adv_predictions = tf.round(tf.nn.sigmoid(adv_logits))
    tp, fp, fn = tp_fp_fn_metrics(labels[:, 1:], adv_predictions[:, 1:])
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    adv_acc = 2 * (pre * rec) / (pre + rec)
  else:
    raise ValueError(FLAGS.attr)

  accs.append(adv_acc)
  opt_loss = loss + gamma * adv_loss

  t_vars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(opt_loss, t_vars)
  grads, variables = zip(*grads_and_vars)
  grads, _ = tf.clip_by_global_norm(grads, 10.0)
  grads_and_vars = zip(grads, variables)
  train_ops = optimizer.apply_gradients(
    grads_and_vars, global_step=global_step)

  iterations = epochs * len(sents) // batch_size
  print_every = FLAGS.print_every
  saver = tf.train.Saver(max_to_keep=epochs)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    iteration = 0
    train_loss = 0
    train_adv_loss = 0
    fw_accs = 0
    bw_accs = 0
    adv_accs = 0

    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
      start = time.time()
      for batch in iterate_triplet_minibatches(sents, sent_masks, authors,
                                               batch_size):
        xx, xp, xr, y = batch
        if FLAGS.attr == 'word':
          b = len(y)
          y = np.zeros((b, vocab_size), dtype=np.float32)
          for i, idx in enumerate(range(b)):
            y[i][xx[0][i]] = 1.0

        feed = {i_inputs: xx[0], i_masks: xx[1],
                p_inputs: xp[0], p_masks: xp[1],
                r_inputs: xr[0], r_masks: xr[1],
                labels: y}

        fetch = sess.run([train_ops, loss, adv_loss] + accs, feed_dict=feed)
        train_loss += fetch[1]
        train_adv_loss += fetch[2]
        fw_accs += fetch[3]
        bw_accs += fetch[4]
        adv_accs += fetch[5]
        iteration += 1
        if iteration % print_every == 0:
          end = time.time()
          log("Iteration: {:.4f}%, Loss: {:.4f}, Adv Loss:{:.4f},"
              " FW Acc:{:.2f}%, BW Acc:{:.2f}%, Adv Perf: {:.2f}%,"
              " {:.4f} sec/batch".format(
                iteration / iterations * 100,
                train_loss / print_every,
                train_adv_loss / print_every,
                fw_accs / print_every * 100,
                bw_accs / print_every * 100,
                adv_accs / print_every * 100,
                (end - start) / print_every))

          train_loss = 0
          train_adv_loss = 0
          fw_accs = 0
          bw_accs = 0
          adv_accs = 0
          start = time.time()

      model_type = FLAGS.cell_type

      if model_type == 'TRANS':
        model_type += 'l{}'.format(FLAGS.num_layer)

      model_name = 'bookcorpus_e{}_{}_b{}_{}_adv{}'.format(
          e, model_type, batch_size, FLAGS.attr, gamma)
      save_path = os.path.join(FLAGS.save_dir, model_name)

      if not os.path.exists(save_path):
        os.makedirs(save_path)

      saver.save(sess, os.path.join(save_path, "model.ckpt"))


if __name__ == '__main__':
  app.run(main)
