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

from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os
import tensorflow as tf
import tqdm
from collections import defaultdict
from keras_applications import resnet

import keras_preprocessing.text

DATA_DIR = '/mnt/nfs/mscoco/'
ANNOTATION_FILE = DATA_DIR + 'annotations/captions_{}2014.json'
IMAGE_DIR = DATA_DIR + '{}2014/'
FEATURE_DIR = DATA_DIR + '{}2014/numpy/'
VOCAB_PATH = DATA_DIR + 'vocab.json'


def ResNet152(*args, **kwargs):
  return resnet.ResNet152(*args, **kwargs)


def resnet_preprocess_input(*args, **kwargs):
  return resnet.preprocess_input(*args, **kwargs)


def read_annotation(mode='train'):
  # Read the json file
  with open(ANNOTATION_FILE.format(mode), 'r') as f:
    annotations = json.load(f)

  # Store captions and image names in vectors
  all_captions = []
  all_img_name_vector = []

  for annot in annotations['annotations']:
    caption = annot['caption']
    image_id = annot['image_id']
    full_coco_image_path = 'COCO_%s2014_%012d.jpg' % (mode, image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

  return all_captions, all_img_name_vector


def get_image_feature_model(name='vgg'):
  if name == 'vgg':
    preprocess_fn = tf.keras.applications.vgg19.preprocess_input
    image_model = tf.keras.applications.vgg19.VGG19(weights='imagenet')
    hidden_layer = image_model.get_layer('fc2').output
  elif name == 'resnet':
    preprocess_fn = resnet_preprocess_input
    image_model = ResNet152(
        weights='imagenet', include_top=False, pooling='avg')
    hidden_layer = image_model.layers[-1].output
  elif name == 'inception':
    preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet', pooling='avg',
        input_shape=(224, 224, 3))
    hidden_layer = image_model.layers[-1].output
  else:
    raise ValueError(name)

  new_input = image_model.input
  image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
  return preprocess_fn, image_features_extract_model


def crop_image(image):
  center_image = tf.image.central_crop(image, central_fraction=0.875)
  cropped_images = [center_image, tf.image.flip_left_right(center_image)]

  for bbox in [(0, 0), (0, 32), (32, 0), (32, 32)]:
    bbox_image = tf.image.crop_to_bounding_box(
        image, bbox[0], bbox[1], 224, 224)
    cropped_images.append(bbox_image)
    cropped_images.append(tf.image.flip_left_right(bbox_image))

  cropped_images = [tf.expand_dims(img, 0) for img in cropped_images]
  return tf.concat(cropped_images, 0)


def extract_inception_features(mode='train', name='inception', batch_size=3,
                               shard=0):
  _, all_img_name_vector = read_annotation(mode)
  preprocess_fn, image_features_extract_model = get_image_feature_model(name)

  def load_image(image_path):
    img = tf.io.read_file(IMAGE_DIR.format(mode) + image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = crop_image(img)
    img = preprocess_fn(img)
    return img, image_path

  # Get unique images
  n_shards = 8
  encode_train = sorted(set(all_img_name_vector))
  n_per_shard = len(encode_train) // 8 + 1
  encode_train = encode_train[shard * n_per_shard: (shard + 1) * n_per_shard]

  # Feel free to change batch_size according to your system configuration
  image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
  image_dataset = image_dataset.map(
      load_image,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)

  feat_dir = os.path.join(FEATURE_DIR.format(mode), name)

  if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

  print('Saving features to', feat_dir)
  pbar = tqdm.tqdm(total=len(encode_train))
  for img, path in image_dataset:
    b = int(img.shape[0])
    img = tf.reshape(img, (b * 10, 224, 224, 3))
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features, (b, 10, -1))
    batch_features = tf.reduce_mean(batch_features, 1)
    for bf, p in zip(batch_features, path):
      path_of_feature = os.path.join(feat_dir, p.numpy().decode("utf-8"))
      np.save(path_of_feature, bf.numpy())
    pbar.update(b)
  pbar.close()


def pad_texts(texts, max_len=50):
  n = len(texts)
  shape = (n, max_len)
  data, masks = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.bool)
  for i, t in enumerate(texts):
    l = min(len(t), max_len)
    data[i, :l] = t[:l]
    masks[i, :l] = 1
  return data, masks


def texts_to_ids(captions, tokenizer, max_len=50):
  captions = tokenizer.texts_to_sequences(captions)
  captions, masks = pad_texts(captions, max_len)
  print(captions.shape, masks.shape)
  return captions, masks


def preprocess_captions(all_captions):
  if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, 'rb') as f:
      tokenizer = keras_preprocessing.text.tokenizer_from_json(f.read())
  else:
    tokenizer = keras_preprocessing.text.Tokenizer(oov_token='<unk>',
                                                   lower=True)
    tokenizer.fit_on_texts(all_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    with open(VOCAB_PATH, 'wb') as f:
      f.write(tokenizer.to_json())

  print('Built vocabulary size:', len(tokenizer.word_index))
  return tokenizer


def images_to_dicts(img_name_vector, mode='train', name='resnet'):
  img_caption_ids = defaultdict(list)
  img_features = dict()

  for i, img_name in enumerate(img_name_vector):
    img_caption_ids[img_name].append(i)

  sorted_img_names = sorted(img_caption_ids.keys())
  for img_name in tqdm.tqdm(sorted_img_names):
    img_features[img_name] = load_feature_vector(img_name, mode, name)
  return img_features, img_caption_ids


def load_feature_vector(img_name, mode='train', name='resnet'):
  feat_path = os.path.join(FEATURE_DIR.format(mode), name, img_name + '.npy')
  return np.load(feat_path)


def load_mscoco_data(name='resnet', train_only=False):
  train_captions, train_img_name_vector = read_annotation('train')
  val_captions, val_img_name_vector = read_annotation('val')
  tokenizer = preprocess_captions(train_captions + val_captions)

  train_captions, train_masks = texts_to_ids(train_captions, tokenizer)
  train_img_features, train_img_caption_ids = \
      images_to_dicts(train_img_name_vector, 'train', name)
  train_data = (train_img_features, train_img_caption_ids,
                train_captions, train_masks)
  if train_only:
    return train_data, tokenizer.word_index

  val_captions, val_masks = texts_to_ids(val_captions, tokenizer)
  val_img_features, val_img_caption_ids = \
      images_to_dicts(val_img_name_vector, 'val', name)
  val_data = (val_img_features, val_img_caption_ids,
              val_captions, val_masks)

  return train_data, val_data, tokenizer.word_index


if __name__ == '__main__':
  # import sys
  # s = int(sys.argv[1])
  # tf.compat.v1.enable_eager_execution()
  # # extract_inception_features('train', shard=s)
  # extract_inception_features('val', shard=s)
  load_mscoco_data()
