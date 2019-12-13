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

from absl import app
from absl import flags

import time
import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import numpy as np
from train_feature_mapper import linear_mapping, gan_mapping, mlp_mapping
from data.bookcorpus import load_author_data as load_bookcorpus_author
from utils.sent_utils import iterate_minibatches_indices, get_similarity_metric
from utils.common_utils import log
from invert.utils import sents_to_labels, tp_fp_fn_metrics_np, sinkhorn

SPM_MODEL_PATH = '/tmp/tfhub_modules/' \
                 '539544f0a997d91c327c23285ea00c37588d92cc/assets/' \
                 'universal_encoder_8k_spm.model'

flags.DEFINE_integer('high_layer_idx', -1, 'Output layer index')
flags.DEFINE_integer('low_layer_idx', -1, 'Optimize layer index')
flags.DEFINE_integer('seq_len', 16, 'Fixed recover sequence length')
flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")
flags.DEFINE_integer('train_size', 250, 'Number of authors data to use')
flags.DEFINE_integer('test_size', 125, 'Number of authors data to test')
flags.DEFINE_integer('max_iters', 1000, 'Max iterations for optimization')
flags.DEFINE_integer('print_every', 1, 'Print metrics every iteration')
flags.DEFINE_integer(
    "max_seq_length", 32,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string('metric', 'l2', 'Metric to optimize')
flags.DEFINE_string('mapper', 'linear', 'Mapper to use')
flags.DEFINE_string('model', 'multiset', 'Model for learning based inversion')
flags.DEFINE_boolean(
  'learning', False,
  'Learning based inversion or optimize based')
flags.DEFINE_boolean(
  'cross_domain', False,
  'Cross domain data for learning based inversion')
flags.DEFINE_float('temp', 0.1, 'Temperature for optimization')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('alpha', 0.0, 'Coefficient for regularization')
flags.DEFINE_float('C', 0.0, 'Label distribution aware margin')
flags.DEFINE_float('wd', 1e-4, 'Weight decay')

FLAGS = flags.FLAGS


LAYER_NAMES = [
  "module_apply_default/Encoder_en/KonaTransformer/Encode/Scatter/ScatterNd:0",
  "module_apply_default/Encoder_en/KonaTransformer/Encode/TransformerStack/"
  "Layer_0/TransformerLayer/FFN/layer_postprocess/add:0",
  "module_apply_default/Encoder_en/KonaTransformer/Encode/TransformerStack/"
  "Layer_1/TransformerLayer/FFN/layer_postprocess/add:0",
  "module_apply_default/Encoder_en/hidden_layers/l2_normalize:0",
]


def sents_to_sparse(ids):
  max_len = max(len(x) for x in ids)
  dense_shape = (len(ids), max_len)
  values = [item for sublist in ids for item in sublist]
  indices = [[row, col] for row in range(len(ids)) for col in
             range(len(ids[row]))]

  return values, indices, dense_shape


def mean_pool(tensor):
  denom = tf.get_default_graph().get_tensor_by_name(
    "module_apply_default/Encoder_en/KonaTransformer/ExpandDims_1:0")
  masks = tf.get_default_graph().get_tensor_by_name(
    "module_apply_default/Encoder_en/KonaTransformer/ExpandDims:0")
  return tf.reduce_sum(tensor * masks, axis=1) / denom


def get_fetch_by_layer(layer_idx):
  fetch = tf.get_default_graph().get_tensor_by_name(LAYER_NAMES[layer_idx])
  if fetch.get_shape().ndims == 3:
    fetch = mean_pool(fetch)
  assert fetch.get_shape().ndims == 2
  return fetch


def load_inversion_data():
  module = hub.Module(
    "https://tfhub.dev/google/universal-sentence-encoder-lite/2")

  sp = spm.SentencePieceProcessor()
  sp.Load(SPM_MODEL_PATH)

  input_placeholder = tf.sparse_placeholder(tf.int64,
                                            shape=[None, None],
                                            name='sparse_placeholder')
  module(inputs=dict(
      values=input_placeholder.values,
      indices=input_placeholder.indices,
      dense_shape=input_placeholder.dense_shape))

  learn_mapping = FLAGS.high_layer_idx != FLAGS.low_layer_idx
  if learn_mapping:
    outputs = [get_fetch_by_layer(FLAGS.low_layer_idx),
               get_fetch_by_layer(FLAGS.high_layer_idx)]
  else:
    outputs = get_fetch_by_layer(FLAGS.low_layer_idx)

  train_sents, _, test_sents, _, _, _ = load_bookcorpus_author(
      train_size=FLAGS.train_size, test_size=FLAGS.test_size,
      unlabeled_size=0, split_by_book=True, split_word=False,
      top_attr=800, remove_punct=False)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  def encode_sents(sents):
    y = [np.asarray(sp.EncodeAsIds(x)[:FLAGS.max_seq_length])
         for x in train_sents]
    y = np.asarray(y)
    n_data = len(sents)
    pbar = tqdm.tqdm(total=n_data)
    embs_low, embs_high = [], []
    for b_idx in iterate_minibatches_indices(n_data, 512):
      values, indices, dense_shape = sents_to_sparse(y[b_idx])
      emb = sess.run(outputs,
                     feed_dict={input_placeholder.values: values,
                                input_placeholder.indices: indices,
                                input_placeholder.dense_shape: dense_shape})
      if learn_mapping:
        embs_low.append(emb[0])
        embs_high.append(emb[1])
      else:
        embs_low.append(emb)
      pbar.update(len(b_idx))

    pbar.close()
    if learn_mapping:
      return [np.vstack(embs_low), np.vstack(embs_high)], y
    else:
      return np.vstack(embs_low), y

  train_x, train_y = encode_sents(train_sents)
  test_x, test_y = encode_sents(test_sents)
  tf.keras.backend.clear_session()

  if learn_mapping:
    log('Training high to low mapping...')
    if FLAGS.mapper == 'linear':
      mapping = linear_mapping(train_x[1], train_x[0])
    elif FLAGS.mapper == 'mlp':
      mapping = mlp_mapping(train_x[1], train_x[0], epochs=10,
                            activation=tf.tanh)
    elif FLAGS.mapper == 'gan':
      mapping = gan_mapping(train_x[1], train_x[0], disc_iters=5,
                            batch_size=64, gamma=1.0, epoch=100,
                            activation=tf.tanh)
    else:
      raise ValueError(FLAGS.mapper)
    test_x = mapping(test_x[1])

  return train_x, train_y, test_x, test_y


def prepare_dummpy_sparse(batch_size, seq_len):
  dense_shape = (batch_size, seq_len)
  ids = np.ones(dense_shape, dtype=np.int64)
  values = [item for sublist in ids for item in sublist]
  indices = [[row, col] for row in range(len(ids)) for col in
             range(len(ids[row]))]
  return values, indices, dense_shape


def replace_graph(emb_lookup, new_emb_lookup):
  names = set()
  for op in tf.get_default_graph().get_operations():
    for index, i in enumerate(op.inputs):
      if i.name == emb_lookup:
        names.add(op.name)
        op._update_input(index, new_emb_lookup)
        break

  # sanity check the inputs has been replaced
  for node in tf.get_default_graph().as_graph_def().node:
    if node.name in names:
      assert new_emb_lookup.name.split(':')[0] in node.input


def optimization_inversion():
  _, _, x, y = load_inversion_data()
  y = sents_to_labels(y)

  max_iters = FLAGS.max_iters
  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len

  embed_module = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
  embed = hub.Module(embed_module)

  sp = spm.SentencePieceProcessor()
  sp.Load(SPM_MODEL_PATH)

  input_placeholder = tf.sparse_placeholder(tf.int64,
                                            shape=[batch_size, None],
                                            name='sparse_placeholder')

  # dummy call to setup the graph
  embed(inputs=dict(values=input_placeholder.values,
                    indices=input_placeholder.indices,
                    dense_shape=input_placeholder.dense_shape))

  emb_lookup = LAYER_NAMES[0]
  start_vars = set(v.name for v in tf.global_variables())

  word_emb = tf.global_variables()[0]

  logit_inputs = tf.get_variable(
    name='logit_inputs',
    shape=(batch_size, seq_len, 8002),
    initializer=tf.random_normal_initializer(-0.1, 0.1))

  permute_inputs = tf.get_variable(
    name='permute_inputs',
    shape=(batch_size, seq_len, seq_len),
    initializer=tf.random_normal_initializer(-0.1, 0.1))
  permute_matrix = sinkhorn(permute_inputs / FLAGS.temp, 10)
  
  prob_inputs = tf.nn.softmax(logit_inputs / FLAGS.temp, axis=-1)
  preds = tf.argmax(prob_inputs, axis=-1)

  emb_inputs = tf.matmul(prob_inputs, word_emb, name='new_embedding_lookup')
  emb_inputs = tf.matmul(permute_matrix, emb_inputs)

  if FLAGS.low_layer_idx == 0:
    encoded = mean_pool(emb_inputs)
  else:
    replace_graph(emb_lookup, emb_inputs)
    encoded = get_fetch_by_layer(FLAGS.low_layer_idx)

  targets = tf.placeholder(tf.float32, name='target',
                           shape=(batch_size, encoded.shape.as_list()[-1]))

  loss = get_similarity_metric(encoded, targets, FLAGS.metric, rtn_loss=True)
  loss = tf.reduce_sum(loss)

  optimizer = tf.train.AdamOptimizer(FLAGS.lr)
  grads_and_vars = optimizer.compute_gradients(loss, [logit_inputs,
                                                      permute_inputs])
  train_ops = optimizer.apply_gradients(
      grads_and_vars, global_step=tf.train.get_or_create_global_step())

  end_vars = tf.global_variables()
  new_vars = [v for v in end_vars if v.name not in start_vars]
  batch_init_ops = tf.variables_initializer(new_vars)

  total_it = len(x) // batch_size

  dummy_inputs = prepare_dummpy_sparse(batch_size, seq_len)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    def invert_one_batch(batch_targets):
      sess.run(batch_init_ops)
      feed_dict = {targets: batch_targets,
                   'sparse_placeholder/values:0': dummy_inputs[0],
                   'sparse_placeholder/indices:0': dummy_inputs[1],
                   'sparse_placeholder/shape:0': dummy_inputs[2]}
      prev = 1e6
      for i in range(max_iters):
        curr, _ = sess.run([loss, train_ops], feed_dict)
        # stop if no progress
        if (i + 1) % (max_iters // 10) == 0 and curr > prev:
          break
        prev = curr
      return sess.run([preds, loss], feed_dict)

    start_time = time.time()
    it = 0.0
    all_tp, all_fp, all_fn, all_err = 0.0, 0.0, 0.0, 0.0

    for batch_idx in iterate_minibatches_indices(len(x), batch_size,
                                                 False, False):
      y_pred, err = invert_one_batch(x[batch_idx])
      tp, fp, fn = tp_fp_fn_metrics_np(y_pred, y[batch_idx])

      it += 1.0
      all_err += err
      all_tp += tp
      all_fp += fp
      all_fn += fn

      all_pre = all_tp / (all_tp + all_fp + 1e-7)
      all_rec = all_tp / (all_tp + all_fn + 1e-7)
      all_f1 = 2 * all_pre * all_rec / (all_pre + all_rec + 1e-7)

      if it % FLAGS.print_every == 0:
        it_time = (time.time() - start_time) / it
        log("Iter {:.2f}%, err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%,"
            " {:.2f} sec/it".format(it / total_it * 100, all_err / it,
                                    all_pre * 100, all_rec * 100,
                                    all_f1 * 100, it_time))

    all_pre = all_tp / (all_tp + all_fp + 1e-7)
    all_rec = all_tp / (all_tp + all_fn + 1e-7)
    all_f1 = 2 * all_pre * all_rec / (all_pre + all_rec + 1e-7)
    log("Final err={}, pre={:.2f}%, rec={:.2f}%, f1={:.2f}%".format(
      all_err / it, all_pre * 100, all_rec * 100, all_f1 * 100))


def main(_):
  optimization_inversion()


if __name__ == '__main__':
  app.run(main)
