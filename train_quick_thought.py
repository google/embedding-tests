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

import tensorflow as tf
import numpy as np
import time
import os
import shutil

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
flags.DEFINE_float('drop_p', 0.15, 'Dropout rate during training')
flags.DEFINE_boolean('scratch', False, 'Train word embedding from scratch')
flags.DEFINE_string('cell_type', 'LSTM', 'Encoder model')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')
FLAGS = flags.FLAGS


def iterate_triplet_minibatches(inputs, masks, batch_size, shuffle=True):
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
    yield batch_inputs, batch_parents, batch_responses


def get_lr(global_step):
  # from attention is all you need
  global_step = tf.cast(global_step, tf.float32)
  arg1 = tf.math.rsqrt(global_step)
  arg2 = global_step * (FLAGS.warmup_steps ** -1.5)
  return tf.math.rsqrt(tf.cast(FLAGS.encoder_dim, tf.float32)) * \
         tf.math.minimum(arg1, arg2)


def main(unused_argv):
  epochs = FLAGS.epochs
  batch_size = FLAGS.batch_size

  sents, sent_masks, vocab = load_bookcorpus_sentences()

  vocab_size = len(vocab) + 1
  log('training with {} sents and {} vocabs'.format(sents.shape, vocab_size))

  if FLAGS.scratch:
    init_word_emb = None
  else:
    init_word_emb = load_initialized_word_emb()
    if init_word_emb.shape[1] < FLAGS.emb_dim:
      offset = FLAGS.emb_dim - init_word_emb.shape[1]
      random_emb = np.random.uniform(-0.1, 0.1, (vocab_size, offset))
      init_word_emb = np.hstack([init_word_emb, random_emb.astype(np.float32)])
    init_word_emb = init_word_emb[:, :FLAGS.emb_dim]

  model = QuickThoughtModel(vocab_size, FLAGS.emb_dim,
                            FLAGS.encoder_dim, FLAGS.context_size,
                            cell_type=FLAGS.cell_type,
                            num_layer=FLAGS.num_layer,
                            init_word_emb=init_word_emb,
                            drop_p=FLAGS.drop_p, train=True)

  global_step = tf.train.get_or_create_global_step()
  lr = get_lr(global_step) if FLAGS.cell_type == 'TRANS' else FLAGS.lr
  optimizer = tf.train.AdamOptimizer(lr)

  i_inputs = tf.placeholder(tf.int64, (None, None), name='i_inputs')
  i_masks = tf.placeholder(tf.int32, (None, None), name='i_masks')
  p_inputs = tf.placeholder(tf.int64, (None, None), name='p_inputs')
  p_masks = tf.placeholder(tf.int32, (None, None), name='p_masks')
  r_inputs = tf.placeholder(tf.int64, (None, None), name='r_inputs')
  r_masks = tf.placeholder(tf.int32, (None, None), name='r_masks')
  output_tensors = [(p_inputs, p_masks), (r_inputs, r_masks)]
  accs, loss = model.forward_triplet((i_inputs, i_masks),
                                     output_tensors, batch_size)

  t_vars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  grads, variables = zip(*grads_and_vars)
  grads, _ = tf.clip_by_global_norm(grads, 10.0)
  grads_and_vars = zip(grads, variables)
  train_ops = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  iterations = epochs * len(sents) // batch_size
  print_every = FLAGS.print_every
  saver = tf.train.Saver(max_to_keep=epochs)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    iteration = 0
    train_loss = 0
    fw_accs = 0
    bw_accs = 0

    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
      start = time.time()
      batches = iterate_triplet_minibatches(sents, sent_masks, batch_size)
      for batch in batches:
        xx, xp, xr = batch
        feed = {i_inputs: xx[0], i_masks: xx[1],
                p_inputs: xp[0], p_masks: xp[1],
                r_inputs: xr[0], r_masks: xr[1]}
        fetch = sess.run([loss, train_ops] + accs, feed_dict=feed)
        train_loss += fetch[0]
        fw_accs += fetch[-2] if len(fetch) == 4 else fetch[-1]
        bw_accs += fetch[-1]

        iteration += 1
        if iteration % print_every == 0:
          end = time.time()
          log('Iteration: {:.4f}%, Loss: {:.4f}, FW Acc:{:.2f}%, '
              'BW Acc:{:.2f}%, {:.4f} sec/batch'.format(
                  iteration / iterations * 100, train_loss / print_every,
                  fw_accs / print_every * 100, bw_accs / print_every * 100,
                  (end - start) / print_every))

          train_loss = 0
          fw_accs = 0
          bw_accs = 0
          start = time.time()

      model_type = FLAGS.cell_type

      if model_type == 'TRANS':
        model_type += 'l{}'.format(FLAGS.num_layer)

      model_name = 'bookcorpus_e{}_{}_b{}'.format(e, model_type, batch_size)
      if FLAGS.scratch:
        model_name += '_scratch'

      save_path = os.path.join(FLAGS.save_dir, model_name)

      if os.path.exists(save_path):
        # remove previously saved model
        shutil.rmtree(save_path)

      if not os.path.exists(save_path):
        os.makedirs(save_path)

      saver.save(sess, os.path.join(save_path, 'model.ckpt'))


if __name__ == '__main__':
  app.run(main)
