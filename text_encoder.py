from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from thought.quick_thought_model import QuickThoughtModel
from utils.sent_utils import iterate_minibatches_indices
from utils.common_utils import make_parallel
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class QuickThoughtEncoder(object):
  def __init__(self, vocab_size, emb_dim=620, encoder_dim=1200, context_size=1,
               cell_type='LSTM', num_layer=3, init_word_emb=None, num_gpu=1):
    self.num_gpu = num_gpu
    self.model = QuickThoughtModel(vocab_size, emb_dim, encoder_dim,
                                   context_size, num_layer=num_layer,
                                   init_word_emb=init_word_emb,
                                   cell_type=cell_type, train=False)

    self.inputs_a = tf.placeholder(tf.int64, (None, None), name="inputs_a")
    self.masks_a = tf.placeholder(tf.int32, (None, None), name="masks_a")

    self.inputs_b = tf.placeholder(tf.int64, (None, None), name="inputs_b")
    self.masks_b = tf.placeholder(tf.int32, (None, None), name="masks_b")

    self.encoded_a = make_parallel(self.encode_fn, self.num_gpu,
                                   x=self.inputs_a, m=self.masks_a)
    self.encoded_b = make_parallel(self.encode_fn, self.num_gpu,
                                   x=self.inputs_b, m=self.masks_b)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

  def load_weights(self, model_path):
    self.saver.restore(self.sess, model_path)

  def encode_fn(self, x, m):
    encode_emb = tf.nn.embedding_lookup(self.model.word_in_emb, x)
    encoded = self.model.encode(encode_emb, m, self.model.in_cells,
                                self.model.proj_in)
    return encoded

  def encode(self, inputs_a, masks_a, query_size):
    feats_a = []
    for idx in iterate_minibatches_indices(len(inputs_a), query_size):
      feed = {self.inputs_a: inputs_a[idx], self.masks_a: masks_a[idx]}
      feat_a = self.sess.run(self.encoded_a, feed_dict=feed)
      feats_a.append(feat_a)

    feats_a = np.vstack(feats_a)
    return feats_a

  def encode_tuple(self, inputs_a, masks_a, inputs_b, masks_b, query_size):
    feats_a, feats_b = [], []
    for idx in iterate_minibatches_indices(len(inputs_a), query_size):
      b = len(idx)
      if b < self.num_gpu:
        continue

      if self.num_gpu > 1:
        offset = b % self.num_gpu
        idx = idx[:b - offset]

      feed = {self.inputs_a: inputs_a[idx], self.masks_a: masks_a[idx],
              self.inputs_b: inputs_b[idx], self.masks_b: masks_b[idx]}

      feat_a, feat_b = self.sess.run([self.encoded_a, self.encoded_b],
                                     feed_dict=feed)
      feats_a.append(feat_a)
      feats_b.append(feat_b)

    feats_a = np.vstack(feats_a)
    feats_b = np.vstack(feats_b)
    return feats_a, feats_b


class UniversalSentenceEncoder(object):
  def __init__(self, inv_vocab):
    import tf_sentencepiece
    embed_module = "https://tfhub.dev/google/universal-sentence-" \
                   "encoder-multilingual-large/1"
    embed = hub.Module(embed_module)
    self.inv_vocab = inv_vocab
    self.inputs = tf.placeholder(tf.string, shape=(None,))
    self.encoded = embed(self.inputs)
    self.sess = tf.Session()
    self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

  def encode(self, sents, masks, query_size=4096):
    sent_embs = []
    for batch_idx in iterate_minibatches_indices(len(sents), query_size,
                                                 shuffle=False):
      x = sents[batch_idx]
      lengths = np.sum(masks[batch_idx], axis=1)
      batch_sents = []

      for sent, l in zip(x, lengths):
        words = [self.inv_vocab.get(w, '<UNK>') for w in sent[:l]]
        batch_sents.append(' '.join(words))

      batch_emb = self.sess.run(self.encoded,
                                feed_dict={self.inputs: batch_sents})
      sent_embs.append(batch_emb)

    sent_embs = np.vstack(sent_embs)
    return sent_embs


class ELMoEncoder(object):
  def __init__(self, inv_vocab):
    embed_module = "https://tfhub.dev/google/elmo/2"
    elmo = hub.Module(embed_module)
    self.inv_vocab = inv_vocab
    self.inputs = tf.placeholder(tf.string, shape=(None,))
    self.encoded = elmo(self.inputs, signature="default",
                        as_dict=True)["default"]
    self.sess = tf.Session()
    self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

  def encode(self, sents, masks, query_size=4096):
    sent_embs = []
    for batch_idx in iterate_minibatches_indices(len(sents), query_size,
                                                 shuffle=False):
      x = sents[batch_idx]
      lengths = np.sum(masks[batch_idx], axis=1)
      batch_sents = []

      for sent, l in zip(x, lengths):
        words = [self.inv_vocab.get(w, '<UNK>') for w in sent[:l]]
        batch_sents.append(' '.join(words))

      batch_emb = self.sess.run(self.encoded,
                                feed_dict={self.inputs: batch_sents})
      sent_embs.append(batch_emb)

    sent_embs = np.vstack(sent_embs)
    return sent_embs


class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids


def convert_single_example(text_a, max_seq_length, tokenizer):
  tokens_a = tokenizer.tokenize(text_a)
  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

  tokens.append("[SEP]")
  segment_ids.append(0)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  return feature


class BERTEncoder(object):
  def __init__(self, inv_vocab):
    from bert import tokenization
    embed_module = "https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1"
    self.bert = hub.Module(embed_module)
    self.inv_vocab = inv_vocab
    self.max_seq_length = 32

    self.inputs = dict(
      input_ids=tf.placeholder(tf.int32, [None, self.max_seq_length],
                               name='input_ids'),
      input_mask=tf.placeholder(tf.int32, [None, self.max_seq_length],
                                name='input_mask'),
      segment_ids=tf.placeholder(tf.int32, [None, self.max_seq_length],
                                 name='segment_ids')
    )

    encoded = self.bert(self.inputs, signature="tokens",
                        as_dict=True)["sequence_output"]
    masks = tf.cast(self.inputs['input_mask'], tf.float32)
    encoded = encoded * tf.expand_dims(masks, axis=2)
    encoded_sum = tf.reduce_sum(encoded, axis=1)
    lengths = tf.reduce_sum(masks, axis=1, keepdims=True)
    self.encoded = encoded_sum / lengths

    self.sess = tf.Session()
    self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    self.tokenizer = self.create_tokenizer_from_hub_module()
    self.tokenization = tokenization

  def create_tokenizer_from_hub_module(self):
    tokenization_info = self.bert(signature="tokenization_info",
                                  as_dict=True)
    vocab_file, do_lower_case = self.sess.run(
      [tokenization_info["vocab_file"],
       tokenization_info["do_lower_case"]]
    )
    return self.tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)

  def encode(self, sents, masks, query_size=4096):
    sent_embs = []
    for batch_idx in iterate_minibatches_indices(len(sents), query_size,
                                                 shuffle=False):
      x = sents[batch_idx]
      lengths = np.sum(masks[batch_idx], axis=1)
      input_ids, input_mask, segment_ids = [], [], []

      for sent, l in zip(x, lengths):
        words = [self.inv_vocab.get(w, '[UNK]') for w in sent[:l]]
        line = ' '.join(words)

        line = self.tokenization.convert_to_unicode(line)
        feature = convert_single_example(line, self.max_seq_length,
                                         self.tokenizer)
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)

      feed_dict = {
        self.inputs['input_ids']: np.asarray(input_ids, dtype=np.int32),
        self.inputs['input_mask']: np.asarray(input_mask, dtype=np.int32),
        self.inputs['segment_ids']: np.asarray(segment_ids, dtype=np.int32),
      }
      batch_emb = self.sess.run(self.encoded, feed_dict=feed_dict)
      sent_embs.append(batch_emb)

    sent_embs = np.vstack(sent_embs)
    return sent_embs


HubEncoders = {'use': UniversalSentenceEncoder,
               'elmo': ELMoEncoder,
               'bert': BERTEncoder}

LocalEncoders = {
  'quickthought': lambda v: QuickThoughtEncoder(vocab_size=v),
  'transformer': lambda v: QuickThoughtEncoder(
    vocab_size=v,  cell_type='TRANS', encoder_dim=600, emb_dim=600,
    num_layer=3),
}
