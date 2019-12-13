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
import time
import tensorflow as tf
import numpy as np
from thought.image_text_model import ImageTextEmbedding
from data.common import get_pretrained_word_embedding, MODEL_DIR
from data.mscoco import load_mscoco_data, DATA_DIR
from utils.sent_utils import iterate_minibatches_indices
from utils.common_utils import log

flags.DEFINE_integer('print_every', 100, 'Print loss every N iterations')
flags.DEFINE_integer('encoder_dim', 512, 'encoder dim')
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_integer('sample_size', 2, 'Sample size of sentence per image')
flags.DEFINE_integer('epochs', 100, 'Epochs of training')
flags.DEFINE_integer('num_neg_sample', 10, 'Negative examples')
flags.DEFINE_float('lr', 1e-4, 'Learning rate for training')
flags.DEFINE_float('margin', 0.2, 'Margin in the loss function')
flags.DEFINE_float('lambda1', 1.0, 'Ratio for image to sentence loss')
flags.DEFINE_float('lambda2', 0.0, 'Ratio for sentence to sentence loss')
flags.DEFINE_float('clip_grad_norm', 10.0, 'Clip gradient norm')
flags.DEFINE_boolean('contrastive', False, 'Use vse++ contrastive loss')
flags.DEFINE_boolean('internal', False, 'Use cross entropy loss')
flags.DEFINE_string('encoder_type', 'rnn', 'Text encoder type')
flags.DEFINE_string('image_model_name', 'resnet',
                    'Image feature extractor name')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 'imgtxt'),
                    'Model directory for embedding model')
FLAGS = flags.FLAGS
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(_):
  epochs = FLAGS.epochs
  batch_size = FLAGS.batch_size
  sample_size = 1 if FLAGS.contrastive else FLAGS.sample_size
  eval_batch_size = 1000

  train_data, val_data, vocab = load_mscoco_data(FLAGS.image_model_name)
  word_embedding = get_pretrained_word_embedding(
      vocab, glove=False, vocab_include_pad=True,
      save_path=DATA_DIR + 'w2v.npz')

  train_img_names = np.sort(train_data[0].keys())
  n_imgs = len(train_img_names)
  iterations = epochs * n_imgs // batch_size

  eval_img_names = np.sort(val_data[0].keys())
  n_eval_imgs = len(eval_img_names)

  img_feat_dim = train_data[0].values()[0].shape[0]
  log('Image feature dimension: {:d}'.format(img_feat_dim))
  feats = tf.placeholder(tf.float32, shape=(None, img_feat_dim), name='feats')
  caps = tf.placeholder(tf.int64, shape=(None, None), name='caps')
  masks = tf.placeholder(tf.bool, shape=(None, None), name='masks')
  labels = tf.placeholder(tf.int64, shape=(None, None), name='labels')
  training = tf.placeholder(tf.bool, shape=(), name='training')

  model = ImageTextEmbedding(word_embedding, FLAGS.encoder_dim,
                             encoder_type=FLAGS.encoder_type, norm=True,
                             drop_p=0.25, margin=FLAGS.margin,
                             num_neg_sample=FLAGS.num_neg_sample,
                             lambda1=FLAGS.lambda1, lambda2=FLAGS.lambda2,
                             contrastive=FLAGS.contrastive,
                             internal=FLAGS.internal)
  loss, recall = model.forward(feats, caps, masks, labels, training)

  lr = FLAGS.lr
  optimizer = tf.train.AdamOptimizer(lr)

  t_vars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  if FLAGS.clip_grad_norm > 0.:
    grads, variables = zip(*grads_and_vars)
    grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_grad_norm)
    grads_and_vars = zip(grads, variables)
  train_ops = optimizer.apply_gradients(
      grads_and_vars, global_step=tf.train.get_or_create_global_step())

  print_every = FLAGS.print_every
  r_to_str = lambda r: '{:.2f}%'.format(r)
  saver = tf.train.Saver(max_to_keep=epochs // 10)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    iteration = 0
    train_loss = 0
    train_recalls = np.zeros(6)

    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
      start = time.time()

      def build_feed_dict(idx, names, data, sz):
        img_features, img_caption_ids, captions, caption_masks = data
        batch_feats = []
        batch_caps = []
        batch_masks = []
        batch_labels = []
        for i, img_name in enumerate(names[idx]):
          batch_feats.append(img_features[img_name])
          text_idx = img_caption_ids[img_name]
          if sz > 0:
            text_idx = np.random.choice(text_idx, sz, replace=False)
          batch_caps.append(captions[text_idx])
          batch_masks.append(caption_masks[text_idx])
          binary_label = np.zeros(len(idx))
          binary_label[i] = 1
          batch_labels.append(np.tile(binary_label, [len(text_idx), 1]))

        return {
            feats: np.vstack(batch_feats), caps: np.vstack(batch_caps),
            masks: np.vstack(batch_masks), labels: np.vstack(batch_labels),
        }

      for batch_idx in iterate_minibatches_indices(n_imgs, batch_size, True,
                                                   False):
        feed_dict = build_feed_dict(batch_idx, train_img_names, train_data,
                                    sample_size)
        feed_dict[training] = True
        fetch = sess.run([loss, recall, train_ops], feed_dict=feed_dict)
        train_loss += fetch[0]
        train_recalls += fetch[1]

        iteration += 1
        if iteration % print_every == 0:
          end = time.time()
          train_recalls = train_recalls / print_every * 100
          train_img_sent_recall = ' '.join(map(r_to_str, train_recalls[:3]))
          train_sent_img_recall = ' '.join(map(r_to_str, train_recalls[3:]))
          log('Iteration: {:.4f}%, Loss: {:.4f},'
              ' I2S Recall: {:s}, S2I Recall: {:s}, '
              ' {:.4f} sec/batch'.format(iteration / iterations * 100,
                                         train_loss / print_every,
                                         train_img_sent_recall,
                                         train_sent_img_recall,
                                         (end - start) / print_every))

          train_loss = 0
          train_recalls = np.zeros(6)
          start = time.time()

      eval_idx = np.random.choice(n_eval_imgs, eval_batch_size, replace=False)
      feed_dict = build_feed_dict(eval_idx, eval_img_names, val_data, 0)
      feed_dict[training] = False
      fetch = sess.run(recall, feed_dict=feed_dict)
      val_recalls = [r * 100 for r in fetch]
      val_img_sent_recall = ' '.join(map(r_to_str, val_recalls[:3]))
      val_sent_img_recall = ' '.join(map(r_to_str, val_recalls[3:]))

      log('\nEpoch: {:d}, I2S Recall: {:s}, S2I Recall: {:s}'.format(
          e + 1, val_img_sent_recall, val_sent_img_recall))

      if (e + 1) % 10 == 0:
        model_name = 'coco_e{}_b{}_{}'.format(
          e + 1, batch_size, FLAGS.encoder_type)
        if FLAGS.contrastive:
          model_name += '_vse'
        elif FLAGS.internal:
          model_name += '_internal'

        save_path = os.path.join(FLAGS.save_dir, model_name)
        if not os.path.exists(save_path):
          os.makedirs(save_path)

        saver.save(sess, os.path.join(save_path, "model.ckpt"))


if __name__ == '__main__':
  app.run(main)
