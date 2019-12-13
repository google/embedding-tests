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
import tqdm
from data.common import EMB_DIR, MODEL_DIR
from data.bookcorpus import load_bookcorpus_sentences, split_bookcorpus, \
    build_vocabulary
from text_encoder import LocalEncoders
from utils.sent_utils import count_rareness, load_raw_sents
from thought import get_model_ckpt_name
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.FATAL)

flags.DEFINE_integer('num_gpu', 1, 'Number of gpus')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('epoch', 0, 'Epochs of training local model')
flags.DEFINE_integer('batch_size', 800, 'Batch size')
flags.DEFINE_integer('freq_min', 90,
                     'use word frequency rank above this percentile, '
                     'e.g. 80=the most infrequent 20 percent words')
flags.DEFINE_integer('query_size', 4096, 'Batch size of query')
flags.DEFINE_integer('min_len', 5, 'Minimum length of text for saving')
flags.DEFINE_float('gamma', 0.0, 'Loss ratio for adversarial')
flags.DEFINE_string('attr', 'author', 'Attributes to censor')
flags.DEFINE_string('data_name', 'bookcorpus', 'Data name')
flags.DEFINE_string('model_name', 'quickthought', 'Model name')
flags.DEFINE_string('model_dir',  os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('emb_dir',  EMB_DIR,
                    'Feature directory for saving embedding model')
flags.DEFINE_boolean('trash_unknown', True, 'Throw sentence with unknown')
flags.DEFINE_boolean('save_context', False, 'Save context for membership '
                                            'inference')
flags.DEFINE_boolean('save_train', False, 'Save embedding for training data')

FLAGS = flags.FLAGS


def filter_sents_indices(sents, masks, freq_threshs):
  lengths = np.sum(masks, axis=1)
  rareness = np.sum(sents, axis=1) / lengths

  cond = (rareness >= freq_threshs[0]) \
      & (rareness < freq_threshs[1]) \
      & (lengths >= FLAGS.min_len)

  rare_sent_idx = np.arange(len(sents))[cond]
  if FLAGS.trash_unknown:
    filtered_indices = []
    for ind in rare_sent_idx:
      length = lengths[ind]
      sent = sents[ind][:length]
      if np.all(sent > 0):
        filtered_indices.append(ind)
    return np.asarray(filtered_indices)
  else:
    return rare_sent_idx


def save_book_raw_sents():
  assert FLAGS.freq_min % 10 == 0
  train_filenames, test_filenames = split_bookcorpus(0)
  train_sents, train_masks, test_sents, test_masks, vocab = \
      load_bookcorpus_sentences(0, test_mi=True)

  print('Counting rareness...')
  freq_threshs = count_rareness(train_sents, train_masks,
                                test_sents, test_masks,
                                percentile=[FLAGS.freq_min,
                                            FLAGS.freq_min + 10])
  if FLAGS.save_context:
    raw_dir = os.path.join(
        FLAGS.emb_dir, 'bookcorpus_raw_rare{}_ctx'.format(FLAGS.freq_min))
  else:
    raw_dir = os.path.join(
        FLAGS.emb_dir, 'bookcorpus_raw_rare{}'.format(FLAGS.freq_min))

  if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

  saved_sents = set()

  def collect_features(sents, masks, filenames):
    indices = np.arange(len(sents))
    np.random.seed(12345)
    np.random.shuffle(indices)

    for i in tqdm.tqdm(indices):
      # filter out some sentences
      rare_sent_idx = filter_sents_indices(sents[i], masks[i], freq_threshs)

      if len(rare_sent_idx) <= 0:
        print('No rare sentences...')
        continue

      valid_sent_idx = []
      for sent_id in rare_sent_idx:
        curr_sent = sents[i][sent_id][: sum(masks[i][sent_id])]
        curr_sent = tuple(curr_sent)
        # remove duplicate
        if curr_sent not in saved_sents:
          if sent_id != len(sents[i]) - 1:
            valid_sent_idx.append(
              sent_id + 1 if FLAGS.save_context else sent_id)

        saved_sents.add(curr_sent)

      if len(valid_sent_idx) <= 0:
        print('No valid sentences...')
        continue

      raw_data_path = os.path.join(raw_dir, filenames[i] + '.npz')
      np.savez(raw_data_path, sents[i][valid_sent_idx],
               masks[i][valid_sent_idx])

  # random shuffle order of saving
  all_sents = train_sents + test_sents
  all_masks = train_masks + test_masks
  all_filenames = train_filenames + test_filenames
  collect_features(all_sents, all_masks, all_filenames)


def save_from_local():
  train_filenames, test_filenames = split_bookcorpus(0)
  all_filenames = train_filenames + test_filenames

  vocab = build_vocabulary(exp_id=0, rebuild=False)
  if FLAGS.save_context:
    sents_dir = os.path.join(
      FLAGS.emb_dir, 'bookcorpus_raw_rare{}_ctx'.format(FLAGS.freq_min))
  else:
    sents_dir = os.path.join(
      FLAGS.emb_dir, 'bookcorpus_raw_rare{}'.format(FLAGS.freq_min))

  if not os.path.exists(sents_dir):
    print('Filtering and saving raw sentence first')
    save_book_raw_sents()

  all_filenames, all_sents, all_masks = load_raw_sents(
    all_filenames, sents_dir, rtn_filenames=True, stack=False)

  vocab_size = len(vocab) + 1
  encoder = LocalEncoders[FLAGS.model_name](vocab_size)
  ckpt_name = get_model_ckpt_name(FLAGS.model_name, epoch=FLAGS.epoch,
                                  batch_size=FLAGS.batch_size,
                                  gamma=FLAGS.gamma, attr=FLAGS.attr)

  model_path = os.path.join(FLAGS.model_dir, ckpt_name, 'model.ckpt')
  encoder.load_weights(model_path)

  if FLAGS.save_context:
    feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}_ctx'.format(
      ckpt_name, FLAGS.freq_min))
  else:
    feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}'.format(
      ckpt_name, FLAGS.freq_min))

  if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

  print('Saving embedding to {}'.format(feat_dir))
  indices = np.arange(len(all_sents))
  for i in tqdm.tqdm(indices):
    embs = encoder.encode(all_sents[i], all_masks[i], FLAGS.query_size)
    np.savez(os.path.join(feat_dir, all_filenames[i] + '.npz'), embs)


def main(_):
  save_from_local()


if __name__ == '__main__':
  app.run(main)
