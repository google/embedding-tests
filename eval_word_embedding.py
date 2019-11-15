from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app

import os
from gensim.models import Word2Vec, FastText
from data.common import MODEL_DIR
from utils.word_utils import load_glove_model, load_tf_embedding

flags.DEFINE_string('model', 'w2v', 'Word embedding model')
flags.DEFINE_float('noise_multiplier', 0.,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 0., 'Clipping norm')
flags.DEFINE_integer('epoch', 4, 'Load model trained this epoch')
flags.DEFINE_integer('microbatches', 128, 'microbatches')
flags.DEFINE_integer('exp_id', 0, 'Experiment trial number')
flags.DEFINE_string('save_dir', os.path.join(MODEL_DIR, 'w2v'),
                    'Model directory for embedding model')
flags.DEFINE_string('eval_data_path', './data/questions-words.txt',
                    'Path for evaluation questions')

FLAGS = flags.FLAGS


def main(unused_argv):
  emb_model = FLAGS.model
  save_dir = FLAGS.save_dir

  model_name = 'wiki9_{}_{}.model'.format(emb_model, FLAGS.exp_id)
  model_path = os.path.join(save_dir, model_name)

  if emb_model == 'ft':
    model = FastText.load(model_path)
  elif emb_model == 'w2v':
    model = Word2Vec.load(model_path)
  elif emb_model == 'glove':
    model = load_glove_model(model_path)
  elif emb_model == 'tfw2v':
    model = load_tf_embedding(FLAGS.exp_id, save_dir=save_dir,
                              epoch=FLAGS.epoch,
                              noise_multiplier=FLAGS.noise_multiplier,
                              l2_norm_clip=FLAGS.l2_norm_clip,
                              microbatches=FLAGS.microbatches)
  else:
    raise ValueError('No such embedding model: {}'.format(emb_model))

  model.accuracy(FLAGS.eval_data_path)


if __name__ == '__main__':
    app.run(main)
