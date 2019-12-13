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
import tqdm
from absl import app
from absl import flags

from data.common import MODEL_DIR, get_pretrained_word_embedding
from data.mscoco import load_mscoco_data, DATA_DIR
from membership.utils import compute_adversarial_advantage
from text_encoder import ImageTextEncoder

flags.DEFINE_integer('print_every', 100, 'Print loss every N iterations')
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_integer('encoder_dim', 512, 'encoder dim')
flags.DEFINE_integer('epoch', 100, 'Epochs of training')
flags.DEFINE_boolean('contrastive', False, 'Use vse++ contrastive loss')
flags.DEFINE_float('train_size', 0.1, 'Temp for sharpening')
flags.DEFINE_string('metric', 'cosine', 'Metric to use for membership')
flags.DEFINE_string('model', 'linear', 'Attack model')
flags.DEFINE_string('encoder_type', 'rnn', 'Text encoder type')
flags.DEFINE_string('image_model_name', 'resnet',
                    'Image feature extractor name')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 'imgtxt'),
                    'Model directory for embedding model')
FLAGS = flags.FLAGS


def collect_metrics(img_embs, text_embs, img_id_text_ids):
  metrics = []
  for img_id in tqdm.tqdm(img_id_text_ids):
    text_ids = img_id_text_ids[img_id]
    i_emb = img_embs[img_id]
    t_embs = text_embs[text_ids]
    if FLAGS.metric == 'l2':
      scores = [-np.linalg.norm(i_emb - t_emb) for t_emb in t_embs]
    elif FLAGS.metric == 'cosine':
      scores = [np.dot(i_emb, t_emb) for t_emb in t_embs]
    else:
      raise ValueError(FLAGS.metrics)

    metrics.append(scores)
  return metrics


def encode_image_text_data():
  train_data, val_data, vocab = load_mscoco_data(FLAGS.image_model_name)
  word_embedding = get_pretrained_word_embedding(
      vocab, glove=False, vocab_include_pad=True,
      save_path=DATA_DIR + 'w2v.npz')
  encoder = ImageTextEncoder(word_embedding, FLAGS.encoder_dim,
                             FLAGS.encoder_type)

  model_name = 'coco_e{}_b{}_{}'.format(
    FLAGS.epoch, FLAGS.batch_size, FLAGS.encoder_type)
  if FLAGS.contrastive:
    model_name += '_vse'

  save_path = os.path.join(FLAGS.save_dir, model_name)

  encoder.load_weights(save_path)
  train_img_embs, train_text_embs, train_img_id_text_ids = \
      encoder.encode(train_data)
  test_img_embs, test_text_embs, test_img_id_text_ids = \
      encoder.encode(val_data)

  train_data = train_img_embs, train_text_embs, train_img_id_text_ids
  test_data = test_img_embs, test_text_embs, test_img_id_text_ids
  tf.keras.backend.clear_session()
  return train_data, test_data


def main(_):
  train_data, test_data = encode_image_text_data()
  train_metrics = collect_metrics(*train_data)
  test_metrics = collect_metrics(*test_data)
  compute_adversarial_advantage([np.mean(m) for m in train_metrics],
                                [np.mean(m) for m in test_metrics])


if __name__ == '__main__':
  app.run(main)
