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

from data.bookcorpus import build_vocabulary
from data.common import MODEL_DIR

from thought.eval import eval_trec, eval_msrp, eval_classification
from thought.vocabulary_expansion import expand_vocabulary
from thought.quick_thought_model import QuickThoughtModel
from utils.sent_utils import group_texts_by_len
import tensorflow as tf
import numpy as np
import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


flags.DEFINE_integer('emb_dim', 620, 'embedding dimension')
flags.DEFINE_integer('encoder_dim', 1200, 'encoder dim')
flags.DEFINE_integer('context_size', 1, 'Context size')
flags.DEFINE_integer('batch_size', 800, 'Batch size')
flags.DEFINE_integer('epoch', 0, 'Epochs of training')
flags.DEFINE_integer('num_layer', 3, 'Number of transformer layer')
flags.DEFINE_string('cell_type', 'LSTM', 'Encoder model')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 's2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('attr', 'author', 'Attributes to censor')
flags.DEFINE_float('gamma', 0., 'Loss ratio for adversarial')
flags.DEFINE_boolean('scratch', False, 'Train word embedding from scratch')
flags.DEFINE_boolean('context', False, 'Negative examples from context or '
                                       'random')

FLAGS = flags.FLAGS


class QuickThoughtEncoder(object):
  def __init__(self, model_path, vocab, emb_dim, encoder_dim, context_size,
               cell_type, num_layer=2):
    vocab, init_word_emb_in = expand_vocabulary(model_path, vocab, "emb_in")

    self.model_path = model_path
    self.vocab = vocab
    vocab_size = len(vocab) + 1

    self.model = QuickThoughtModel(vocab_size, emb_dim, encoder_dim,
                                   context_size, num_layer=num_layer,
                                   init_word_emb=None, cell_type=cell_type,
                                   train=False)

    self.inputs = tf.placeholder(tf.int64, (None, None), name='inputs')
    self.masks = tf.placeholder(tf.int32, (None, None), name='masks')

    encode_emb_a = tf.nn.embedding_lookup(self.model.word_in_emb, self.inputs)
    encoded_a = self.model.encode(encode_emb_a, self.masks,
                                  self.model.in_cells, self.model.proj_in)
    encoded_norm_a = tf.nn.l2_normalize(encoded_a, axis=-1)
    self.encoded = encoded_a
    self.encoded_norm = encoded_norm_a

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    # assign expanded embedding
    emb_plhdr = tf.placeholder(tf.float32, shape=(vocab_size, emb_dim))
    self.sess.run(self.model.word_in_emb.assign(emb_plhdr),
                  {emb_plhdr: init_word_emb_in})
    var_list = {v.name[:-2]: v
                for v in tf.global_variables() if not v.name.startswith('emb')}
    self.saver = tf.train.Saver(var_list)

  def load_weights(self):
    print('Loading weight from {}'.format(self.model_path))
    self.saver.restore(self.sess, os.path.join(self.model_path, 'model.ckpt'))

  def preprocess_data(self, texts):
    data = []
    for text in texts:
      sent = [self.vocab.get(word, 0) for word in text.split()]
      data.append(np.asarray(sent, dtype=np.int))
    return data

  def encode(self, texts, verbose=True, norm=False):
    data = self.preprocess_data(texts)
    feats = []
    batches, batch_indices = group_texts_by_len(data)
    n = len(batches)

    indices = []
    for i in tqdm.trange(n) if verbose else range(n):
      inputs = np.asarray(batches[i])
      indices += batch_indices[i]
      masks = np.ones_like(inputs).astype(np.int32)
      feed = {self.inputs: inputs, self.masks: masks}
      fetch = self.encoded_norm if norm else self.encoded
      feat = self.sess.run(fetch, feed_dict=feed)
      feats.append(feat)

    assert len(indices) == len(data)
    feats = np.vstack(feats)
    results = np.zeros_like(feats)
    results[indices] = feats
    return results


def main(_):
  vocab = build_vocabulary(0, rebuild=False)
  model_type = FLAGS.cell_type
  if model_type == 'TRANS':
    model_type += 'l{}'.format(FLAGS.num_layer)

  if FLAGS.gamma == 0.:
    model_name = 'bookcorpus_e{}_{}_b{}'.format(
        FLAGS.epoch, model_type, FLAGS.batch_size)
    if FLAGS.scratch:
      model_name += '_scratch'
    if FLAGS.context:
      model_name += '_context'
  else:
    model_name = 'bookcorpus_e{}_{}_b{}_{}_adv{}'.format(
        FLAGS.epoch, model_type, FLAGS.batch_size, FLAGS.attr, FLAGS.gamma)
  model_path = os.path.join(FLAGS.save_dir, model_name)

  encoder = QuickThoughtEncoder(model_path, vocab, FLAGS.emb_dim,
                                FLAGS.encoder_dim, FLAGS.context_size,
                                FLAGS.cell_type, num_layer=FLAGS.num_layer)
  encoder.load_weights()

  all_scores = dict()

  all_scores['MSRP'] = eval_msrp.evaluate(encoder)[1]
  all_scores['TREC'] = eval_trec.evaluate(encoder)

  for dataset in ['SUBJ', 'MPQA']:
    all_scores[dataset] = eval_classification.eval_nested_kfold(encoder,
                                                                dataset)

  result_path = './downstream_result/'
  if not os.path.exists(result_path):
    os.makedirs(result_path)

  with open(os.path.join(result_path, model_name), 'w') as f:
    for k, v in all_scores.items():
      f.write('{}: {}\n'.format(k, v))


if __name__ == '__main__':
    app.run(main)
