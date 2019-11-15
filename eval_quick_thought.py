from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

import os

from data.bookcorpus import build_vocabulary
from data.common import MODEL_DIR
from thought import eval_trec, eval_msrp
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
flags.DEFINE_integer('batch_size', 500, 'Batch size')
flags.DEFINE_integer('epoch', 0, 'Epochs of training')
flags.DEFINE_integer('num_layer', 3, 'Number of transformer layer')
flags.DEFINE_string('cell_type', 'LSTM', 'Encoder model')
flags.DEFINE_string('eval_data', 'trec', 'eval dataset')
flags.DEFINE_string('save_dir', MODEL_DIR,
                    'Model directory for embedding model')
flags.DEFINE_float('gamma', 0., 'Loss ratio for adversarial')
flags.DEFINE_boolean('scratch', False, 'Train word embedding from scratch')

FLAGS = flags.FLAGS


class QuickThoughtEncoder(object):
  def __init__(self, model_path, vocab, emb_dim, encoder_dim, context_size,
               cell_type, num_layer=2):
    vocab, init_word_emb = expand_vocabulary(model_path, vocab)
    self.model_path = model_path
    self.vocab = vocab
    vocab_size = len(vocab) + 1

    self.model = QuickThoughtModel(vocab_size, emb_dim, encoder_dim,
                                   context_size, num_layer=num_layer,
                                   init_word_emb=None, cell_type=cell_type,
                                   train=False)

    self.inputs = tf.placeholder(tf.int64, (None, None), name='inputs')
    self.masks = tf.placeholder(tf.int32, (None, None), name='masks')
    encode_in_emb = tf.nn.embedding_lookup(self.model.word_in_emb, self.inputs)
    self.encoded = self.model.encode(encode_in_emb, self.masks,
                                     self.model.in_cells, self.model.proj_in)
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    # assign expanded embedding
    emb_plhdr = tf.placeholder(tf.float32, shape=(vocab_size, emb_dim))
    self.sess.run(self.model.word_in_emb.assign(emb_plhdr),
                  {emb_plhdr: init_word_emb})

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

  def encode(self, texts, verbose=True):
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
      feat = self.sess.run(self.encoded, feed_dict=feed)
      feats.append(feat)

    assert len(indices) == len(data)
    feats = np.vstack(feats)
    results = np.zeros_like(feats)
    results[indices] = feats
    return results


def main(unused_argv):
  vocab = build_vocabulary(0, rebuild=False)
  model_type = FLAGS.cell_type
  if model_type == 'TRANS':
    model_type += 'l{}'.format(FLAGS.num_layer)

  if FLAGS.gamma == 0.:
    model_name = 'bookcorpus_e{}_{}_b{}'.format(
        FLAGS.epoch, model_type, FLAGS.batch_size)
    if FLAGS.scratch:
      model_name += '_scratch'
  else:
    model_name = 'bookcorpus_e{}_{}_b{}_adv{}'.format(
        FLAGS.epoch, model_type, FLAGS.batch_size, FLAGS.gamma)
  model_path = os.path.join(FLAGS.save_dir, model_name)

  encoder = QuickThoughtEncoder(model_path, vocab, FLAGS.emb_dim,
                                FLAGS.encoder_dim, FLAGS.context_size,
                                FLAGS.cell_type, num_layer=FLAGS.num_layer)
  encoder.load_weights()

  if FLAGS.eval_data == 'trec':
    eval_trec.evaluate(encoder)
  if FLAGS.eval_data == 'msrp':
    eval_msrp.evaluate(encoder)


if __name__ == '__main__':
    app.run(main)
