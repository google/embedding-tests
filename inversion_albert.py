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

from data.common import MODEL_DIR
from data.wiki103 import load_wiki_cross_domain_data as load_cross_domain_data
from data.bookcorpus import load_author_data as load_bookcorpus_author
from utils.sent_utils import iterate_minibatches_indices, get_similarity_metric
from utils.common_utils import log
from invert.bert_common import read_examples, convert_examples_to_features,\
  mean_pool
from invert.models import MultiLabelInversionModel, MultiSetInversionModel
from invert.utils import count_label_freq, tp_fp_fn_metrics_np, \
  tp_fp_fn_metrics
from train_feature_mapper import gan_mapping, mlp_mapping, linear_mapping
import os
import numpy as np
import tqdm
import models.albert.modeling as modeling
import models.albert.tokenization as tokenization
import tensorflow as tf
import time

ALBERT_DIR = os.path.join(MODEL_DIR, 'albert', 'albert_base')

flags.DEFINE_integer('high_layer_idx', -1, 'Output layer index')
flags.DEFINE_integer('low_layer_idx', -1, 'Optimize layer index')
flags.DEFINE_integer('seq_len', 16, 'Fixed recover sequence length')
flags.DEFINE_integer("batch_size", 16, "Batch size for predictions.")
flags.DEFINE_integer('train_size', 250, 'Number of authors data to use')
flags.DEFINE_integer('test_size', 125, 'Number of authors data to test')
flags.DEFINE_integer('max_iters', 1000, 'Max iterations for optimization')
flags.DEFINE_integer('print_every', 1, 'Print metrics every iteration')
flags.DEFINE_integer(
    "max_seq_length", 32,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string(
    "init_checkpoint", os.path.join(ALBERT_DIR, 'model.ckpt-best'),
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string(
    "albert_config_file", os.path.join(ALBERT_DIR, 'albert_config.json'),
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string(
    "vocab_file", os.path.join(ALBERT_DIR, '30k-clean.vocab'),
    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "spm_model_file",  os.path.join(ALBERT_DIR, '30k-clean.model'),
    "The model file for sentence piece tokenization.")
flags.DEFINE_string('metric', 'l2', 'Metric to optimize')
flags.DEFINE_string('mapper', 'linear', 'Mapper to use')
flags.DEFINE_string('model', 'multiset', 'Model for learning based inversion')
flags.DEFINE_boolean(
  'learning', False,
  'Learning based inversion or optimize based')
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_cls_token", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_boolean(
  'cross_domain', False,
  'Cross domain data for learning based inversion')
flags.DEFINE_float('temp', 1e-2, 'Temperature for optimization')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('alpha', 0.0, 'Coefficient for regularization')
flags.DEFINE_float('C', 0.0, 'Label distribution aware margin')
flags.DEFINE_float('wd', 1e-4, 'Weight decay')
FLAGS = flags.FLAGS


def postprocess_last_layer(input_tensor, config):
  with tf.variable_scope('cls/predictions', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('transform', reuse=tf.AUTO_REUSE):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=config.embedding_size,
          activation=modeling.get_activation(config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)
  return input_tensor


def model_fn_builder(albert_config, init_checkpoint, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length],
                             name='input_ids')
  input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length],
                              name='input_mask')
  input_type_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length],
                                  name='segment_ids')
  model = modeling.AlbertModel(
      config=albert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=input_type_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  all_layer_outputs = [model.get_word_embedding_output()]
  all_layer_outputs += model.get_all_encoder_layers()
  if FLAGS.high_layer_idx == FLAGS.low_layer_idx:
    if FLAGS.use_cls_token:
      outputs = model.get_pooled_output()
    else:
      outputs = all_layer_outputs[FLAGS.high_layer_idx]
      outputs = mean_pool(outputs, input_mask)
  else:
    low_outputs = all_layer_outputs[FLAGS.low_layer_idx]
    low_outputs = mean_pool(low_outputs, input_mask)
    if FLAGS.use_cls_token:
      high_outputs = model.get_pooled_output()
    else:
      high_outputs = all_layer_outputs[FLAGS.high_layer_idx]
      high_outputs = mean_pool(high_outputs, input_mask)
    outputs = (low_outputs, high_outputs)

  tvars = tf.trainable_variables()
  (assignment_map,
   initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
       tvars, init_checkpoint)

  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
  return input_ids, input_mask, input_type_ids, outputs


def load_inversion_data():
  albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file
  )

  train_sents, _, test_sents, _, _, _ = load_bookcorpus_author(
      train_size=FLAGS.train_size, test_size=FLAGS.test_size,
      unlabeled_size=0, split_by_book=True, split_word=False,
      top_attr=800)

  if FLAGS.cross_domain:
    train_sents = load_cross_domain_data(800000, split_word=False)

  def sents_to_examples(sents):
    examples = read_examples(sents, tokenization.convert_to_unicode)
    return convert_examples_to_features(examples=examples,
                                        seq_length=FLAGS.max_seq_length,
                                        tokenizer=tokenizer)

  input_ids, input_mask, input_type_ids, outputs = model_fn_builder(
      albert_config=albert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_one_hot_embeddings=False)

  sess = tf.Session()
  sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
  learn_mapping = FLAGS.high_layer_idx != FLAGS.low_layer_idx

  def encode_example(features):
    n_data = len(features[0])
    embs_low, embs_high = [], []
    pbar = tqdm.tqdm(total=n_data)
    for b_idx in iterate_minibatches_indices(n_data, 128):
      emb = sess.run(outputs, feed_dict={input_ids: features[0][b_idx],
                                         input_mask: features[1][b_idx],
                                         input_type_ids: features[2][b_idx]})
      if learn_mapping:
        embs_low.append(emb[0])
        embs_high.append(emb[1])
        n_batch = len(emb[0])
      else:
        embs_low.append(emb)
        n_batch = len(emb)
      pbar.update(n_batch)
    pbar.close()

    if learn_mapping:
      return np.vstack(embs_low), np.vstack(embs_high)
    else:
      return np.vstack(embs_low)

  train_features = sents_to_examples(train_sents)
  train_x = encode_example(train_features)

  test_features = sents_to_examples(test_sents)
  test_x = encode_example(test_features)
  tf.keras.backend.clear_session()

  if learn_mapping:
    log('Training high to low mapping...')
    if FLAGS.mapper == 'linear':
      mapping = linear_mapping(train_x[1], train_x[0])
    elif FLAGS.mapper == 'mlp':
      mapping = mlp_mapping(train_x[1], train_x[0], epochs=50,
                            activation=modeling.gelu)
    elif FLAGS.mapper == 'gan':
      mapping = gan_mapping(train_x[1], train_x[0], disc_iters=5,
                            batch_size=64, gamma=1.0, epoch=100,
                            activation=tf.tanh)
    else:
      raise ValueError(FLAGS.mapper)
    test_x = mapping(test_x[1])

  return train_x, train_features, test_x, test_features


def encode(embedding_output, input_mask, token_type_ids, config):
  with tf.variable_scope("bert", reuse=True):
    with tf.variable_scope("embeddings", reuse=True):
      embedding_output = modeling.embedding_postprocessor(
          input_tensor=embedding_output,
          use_token_type=True,
          token_type_ids=token_type_ids,
          token_type_vocab_size=config.type_vocab_size,
          token_type_embedding_name='token_type_embeddings',
          use_position_embeddings=True,
          position_embedding_name='position_embeddings',
          initializer_range=config.initializer_range,
          max_position_embeddings=config.max_position_embeddings,
          dropout_prob=config.hidden_dropout_prob)

    with tf.variable_scope('encoder', reuse=True):
      all_encoder_layers = modeling.transformer_model(
          input_tensor=embedding_output,
          attention_mask=input_mask,
          hidden_size=config.hidden_size,
          num_hidden_layers=config.num_hidden_layers,
          num_hidden_groups=config.num_hidden_groups,
          num_attention_heads=config.num_attention_heads,
          intermediate_size=config.intermediate_size,
          inner_group_num=config.inner_group_num,
          intermediate_act_fn=modeling.get_activation(config.hidden_act),
          hidden_dropout_prob=config.hidden_dropout_prob,
          attention_probs_dropout_prob=config.attention_probs_dropout_prob,
          initializer_range=config.initializer_range,
          do_return_all_layers=True)

    all_encoder_layers = [embedding_output] + all_encoder_layers
    sequence_output = all_encoder_layers[FLAGS.low_layer_idx]
    if FLAGS.use_cls_token:
      with tf.variable_scope("pooler", reuse=True):
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=modeling.create_initializer(
              config.initializer_range))
    else:
      pooled_output = mean_pool(sequence_output, input_mask)
  return pooled_output


def filter_labels(labels, filters):
  new_labels = []
  for y in labels:
    new_y = np.setdiff1d(y, filters)
    new_labels.append(new_y)
  return np.asarray(new_labels)


def optimization_inversion():
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file)
  cls_id = tokenizer.vocab['[CLS]']
  sep_id = tokenizer.vocab['[SEP]']
  mask_id = tokenizer.vocab['[MASK]']

  _, _, x, y = load_inversion_data()
  y = y[0]
  filters = [cls_id, sep_id, mask_id]
  y = filter_labels(y, filters)

  batch_size = FLAGS.batch_size
  seq_len = FLAGS.seq_len
  max_iters = FLAGS.max_iters

  albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)
  input_ids = tf.ones((batch_size, seq_len + 2), tf.int32)
  input_mask = tf.ones_like(input_ids, tf.int32)
  input_type_ids = tf.zeros_like(input_ids, tf.int32)

  model = modeling.AlbertModel(
      config=albert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=input_type_ids,
      use_one_hot_embeddings=False)

  word_emb = model.output_embedding_table

  albert_vars = tf.trainable_variables()
  (assignment_map,
   _) = modeling.get_assignment_map_from_checkpoint(albert_vars,
                                                    FLAGS.init_checkpoint)
  tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

  batch_cls_ids = tf.ones((batch_size, 1), tf.int32) * cls_id
  batch_sep_ids = tf.ones((batch_size, 1), tf.int32) * sep_id
  cls_emb = tf.nn.embedding_lookup(word_emb, batch_cls_ids)
  sep_emb = tf.nn.embedding_lookup(word_emb, batch_sep_ids)

  prob_mask = np.zeros((albert_config.vocab_size,), np.float32)
  prob_mask[filters] = -1e9
  prob_mask = tf.constant(prob_mask, dtype=np.float32)

  logit_inputs = tf.get_variable(
      name='inputs',
      shape=(batch_size, seq_len, albert_config.vocab_size),
      initializer=tf.random_uniform_initializer(-0.1, 0.1))

  t_vars = [logit_inputs]
  t_var_names = {logit_inputs.name}

  logit_inputs += prob_mask
  prob_inputs = tf.nn.softmax(logit_inputs / FLAGS.temp, axis=-1)

  emb_inputs = tf.matmul(prob_inputs, word_emb)
  emb_inputs = tf.concat([cls_emb, emb_inputs, sep_emb], axis=1)

  if FLAGS.low_layer_idx == 0:
    encoded = mean_pool(emb_inputs, input_mask)
  else:
    encoded = encode(emb_inputs, input_mask, input_type_ids, albert_config)
  targets = tf.placeholder(
        tf.float32, shape=(batch_size, encoded.shape.as_list()[-1]))

  loss = get_similarity_metric(encoded, targets, FLAGS.metric, rtn_loss=True)
  loss = tf.reduce_sum(loss)

  optimizer = tf.train.AdamOptimizer(FLAGS.lr)

  start_vars = set(v.name for v in tf.global_variables()
                   if v.name not in t_var_names)
  grads_and_vars = optimizer.compute_gradients(loss, t_vars)
  train_ops = optimizer.apply_gradients(
      grads_and_vars, global_step=tf.train.get_or_create_global_step())

  end_vars = tf.global_variables()
  new_vars = [v for v in end_vars if v.name not in start_vars]

  preds = tf.argmax(prob_inputs, axis=-1)
  batch_init_ops = tf.variables_initializer(new_vars)

  total_it = len(x) // batch_size
  with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def invert_one_batch(batch_targets):
      sess.run(batch_init_ops)
      feed_dict = {targets: batch_targets}
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
      # for yp, yt in zip(y_pred, y[batch_idx]):
      #   print(' '.join(set(tokenizer.convert_ids_to_tokens(yp))))
      #   print(' '.join(set(tokenizer.convert_ids_to_tokens(yt))))

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


def learning_inversion():
  assert FLAGS.low_layer_idx == FLAGS.high_layer_idx == -1

  albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)
  num_words = albert_config.vocab_size

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file)

  cls_id = tokenizer.vocab['[CLS]']
  sep_id = tokenizer.vocab['[SEP]']
  mask_id = tokenizer.vocab['[MASK]']

  train_x, train_y, test_x, test_y = load_inversion_data()
  filters = [cls_id, sep_id, mask_id, 0]
  train_y = filter_labels(train_y[0], filters)
  test_y = filter_labels(test_y[0], filters)

  label_freq = count_label_freq(train_y, num_words)
  log('Imbalace ratio: {}'.format(np.max(label_freq) / np.min(label_freq)))

  label_margin = tf.constant(np.reciprocal(label_freq ** 0.25),
                             dtype=tf.float32)
  C = FLAGS.C

  log('Build attack model for {} words...'.format(num_words))

  encoder_dim = train_x.shape[1]
  inputs = tf.placeholder(tf.float32, (None, encoder_dim), name="inputs")
  labels = tf.placeholder(tf.float32, (None, num_words), name="labels")
  training = tf.placeholder(tf.bool, name='training')

  if FLAGS.model == 'multiset':
    emb_dim = 512
    model = MultiSetInversionModel(emb_dim, num_words, FLAGS.seq_len, None,
                                   C=C, label_margin=label_margin)
  elif FLAGS.model == 'multilabel':
    model = MultiLabelInversionModel(num_words, C=C, label_margin=label_margin)
  else:
    raise ValueError(FLAGS.model)

  preds, loss = model.forward(inputs, labels, training)
  true_pos, false_pos, false_neg = tp_fp_fn_metrics(labels, preds)
  eval_fetch = [loss, true_pos, false_pos, false_neg]

  t_vars = tf.trainable_variables()
  wd = FLAGS.wd
  post_ops = [tf.assign(v, v * (1 - wd)) for v in t_vars if 'kernel' in v.name]

  optimizer = tf.train.AdamOptimizer(FLAGS.lr)
  grads_and_vars = optimizer.compute_gradients(
    loss + tf.losses.get_regularization_loss(), t_vars)
  train_ops = optimizer.apply_gradients(
    grads_and_vars, global_step=tf.train.get_or_create_global_step())

  with tf.control_dependencies([train_ops]):
    train_ops = tf.group(*post_ops)

  log('Train attack model with {} data...'.format(len(train_x)))
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(30):
      train_iterations = 0
      train_loss = 0

      for batch_idx in iterate_minibatches_indices(len(train_y),
                                                   FLAGS.batch_size, True):
        one_hot_labels = np.zeros((len(batch_idx), num_words),
                                  dtype=np.float32)
        for i, idx in enumerate(batch_idx):
          one_hot_labels[i][train_y[idx]] = 1
        feed = {inputs: train_x[batch_idx], labels: one_hot_labels,
                training: True}
        err, _ = sess.run([loss, train_ops], feed_dict=feed)
        train_loss += err
        train_iterations += 1

      test_iterations = 0
      test_loss = 0
      test_tp, test_fp, test_fn = 0, 0, 0

      for batch_idx in iterate_minibatches_indices(len(test_y), batch_size=512,
                                                   shuffle=False):
        one_hot_labels = np.zeros((len(batch_idx), num_words),
                                  dtype=np.float32)
        for i, idx in enumerate(batch_idx):
          one_hot_labels[i][test_y[idx]] = 1
        feed = {inputs: test_x[batch_idx], labels: one_hot_labels,
                training: False}

        fetch = sess.run(eval_fetch, feed_dict=feed)
        err, tp, fp, fn = fetch

        test_iterations += 1
        test_loss += err
        test_tp += tp
        test_fp += fp
        test_fn += fn

      precision = test_tp / (test_tp + test_fp) * 100
      recall = test_tp / (test_tp + test_fn) * 100
      f1 = 2 * precision * recall / (precision + recall)

      log("Epoch: {}, train loss: {:.4f}, test loss: {:.4f}, "
          "pre: {:.2f}%, rec: {:.2f}%, f1: {:.2f}%".format(
            epoch, train_loss / train_iterations,
            test_loss / test_iterations,
            precision, recall, f1))


def main(_):
  if FLAGS.learning:
    assert FLAGS.low_layer_idx == FLAGS.high_layer_idx
    learning_inversion()
  else:
    optimization_inversion()


if __name__ == '__main__':
  app.run(main)
