from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

import os
import tqdm
from data.common import MODEL_DIR, EMB_DIR, DATA_DIR
from data.bookcorpus import load_bookcorpus_sentences, split_bookcorpus, \
    build_vocabulary
from data.wiki103 import NUM_SHARD
from text_encoder import LocalEncoders, HubEncoders
from utils.sent_utils import count_rareness, load_raw_sents
from thought import ThoughtModelNameFunc
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.FATAL)

flags.DEFINE_integer('num_gpu', 1, 'Number of gpus')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('epoch', 0, 'Epochs of training local model')
flags.DEFINE_integer('freq_min', 90,
                     'use word frequency rank above this percentile, '
                     'e.g. 80=the most infrequent 20 percent words')
flags.DEFINE_integer('query_size', 4096, 'Batch size of query')
flags.DEFINE_integer('min_len', 5, 'Minimum length of text for saving')
flags.DEFINE_string('cell_type', 'LSTM', 'Encoder model')
flags.DEFINE_string('model_name', 'quickthought', 'Model name')
flags.DEFINE_string('model_dir',  MODEL_DIR,
                    'Model directory for embedding model')
flags.DEFINE_string('emb_dir', EMB_DIR,
                    'Feature directory for saving embedding model')
flags.DEFINE_boolean('trash_unknown', True, 'Throw sentence with unknown')
flags.DEFINE_boolean('save_context', False, 'Save context for membership '
                                            'inference')
flags.DEFINE_boolean('save_cross_domain', False,
                     'Save cross domain data for training inversion model')
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


def save_raw_sents():
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
    save_raw_sents()

  all_filenames, all_sents, all_masks = load_raw_sents(
    all_filenames, sents_dir, rtn_filenames=True, stack=False)

  vocab_size = len(vocab) + 1
  encoder = LocalEncoders[FLAGS.model_name](vocab_size)
  model_name = ThoughtModelNameFunc[FLAGS.model_name](FLAGS.epoch)

  model_path = os.path.join(FLAGS.model_dir, model_name, 'model.ckpt')
  encoder.load_weights(model_path)

  if FLAGS.save_context:
    feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}_ctx'.format(
      model_name, FLAGS.freq_min))
  else:
    feat_dir = os.path.join(FLAGS.emb_dir, '{}_rare{}'.format(
      model_name, FLAGS.freq_min))

  if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

  print('Saving embedding to {}'.format(feat_dir))
  indices = np.arange(len(all_sents))
  for i in tqdm.tqdm(indices):
    embs = encoder.encode(all_sents[i], all_masks[i], FLAGS.query_size)
    np.savez(os.path.join(feat_dir, all_filenames[i] + '.npz'), embs)


def save_from_hub():
  _, test_filenames = split_bookcorpus(0)
  all_filenames = test_filenames

  vocab = build_vocabulary(exp_id=0, rebuild=False)
  inv_vocab = dict((v, k) for k, v in vocab.items())

  sents_dir = os.path.join(FLAGS.emb_dir,
                           'bookcorpus_raw_rare{}'.format(FLAGS.freq_min))
  fnames, all_sents, all_masks = load_raw_sents(
    all_filenames, sents_dir, rtn_filenames=True, stack=False)

  model_name = FLAGS.model_name
  feat_dir = os.path.join(FLAGS.emb_dir, 'bookcorpus_{}_rare{}'.format(
    model_name, FLAGS.freq_min))

  if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

  encoder = HubEncoders[model_name](inv_vocab)

  all_gpus = FLAGS.num_gpu
  indices = np.arange(len(fnames))
  shards_per_gpu = len(fnames) // all_gpus + 1

  indices = indices[FLAGS.gpu_id * shards_per_gpu:
                    (FLAGS.gpu_id + 1) * shards_per_gpu]

  for i in tqdm.tqdm(indices):
    sents, masks = all_sents[i], all_masks[i]
    sent_embs = encoder.encode(sents, masks, FLAGS.query_size)
    save_path = os.path.join(feat_dir, fnames[i] + '.npz')
    np.savez(save_path, sent_embs)


def save_cross_domain_embedding():
  model_name = FLAGS.model_name
  if model_name in HubEncoders:
    vocab = build_vocabulary(exp_id=0, rebuild=False)
    inv_vocab = dict((v, k) for k, v in vocab.items())
    encoder = HubEncoders[model_name](inv_vocab)
  else:
    assert model_name in LocalEncoders
    encoder = LocalEncoders[FLAGS.model_name](50001)
    ckpt_name = ThoughtModelNameFunc[FLAGS.model_name](FLAGS.epoch)
    encoder.load_weights(
      os.path.join(FLAGS.model_dir, ckpt_name, 'model.ckpt'))

  emb_dir = os.path.join(DATA_DIR, 'wiki103', model_name)
  if not os.path.exists(emb_dir):
    os.makedirs(emb_dir)

  def save_embs(cross_domain_sents, cross_domain_masks, name='wiki'):
    n = len(cross_domain_sents)
    shard_size = n // NUM_SHARD + 1
    for i in tqdm.trange(NUM_SHARD):
      sents = cross_domain_sents[i * shard_size: (i + 1) * shard_size]
      masks = cross_domain_masks[i * shard_size: (i + 1) * shard_size]
      embs = encoder.encode(sents, masks, query_size=FLAGS.query_size)
      np.savez(os.path.join(emb_dir, '{}_emb{}-{}.npz'.format(
        name, i + 1, NUM_SHARD)), embs)

  if FLAGS.save_context:
    save_path = os.path.join(DATA_DIR, 'wiki103', 'wiki_invert_ctx.npz')
    with np.load(save_path) as f:
      cross_domain_sents_a, cross_domain_masks_a = f['arr_0'], f['arr_1']
      cross_domain_sents_b, cross_domain_masks_b = f['arr_2'], f['arr_3']
    print('Saving {} sents, {} contexts'.format(len(cross_domain_sents_a),
                                                len(cross_domain_sents_b)))
    save_embs(cross_domain_sents_a, cross_domain_masks_a, name='wiki_ctx_a')
    save_embs(cross_domain_sents_b, cross_domain_masks_b, name='wiki_ctx_b')
  else:
    save_path = os.path.join(DATA_DIR, 'wiki103', 'wiki_invert.npz')
    with np.load(save_path) as f:
      cross_domain_sents_a, cross_domain_masks_a = f['arr_0'], f['arr_1']
    save_embs(cross_domain_sents_a, cross_domain_masks_a)


def main(unused_argv):
  if FLAGS.save_cross_domain:
    save_cross_domain_embedding()
  elif FLAGS.model_name in HubEncoders:
    save_from_hub()
  else:
    assert FLAGS.model_name in ThoughtModelNameFunc
    save_from_local()


if __name__ == '__main__':
  app.run(main)
