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

import sys
import tensorflow as tf
from contextlib2 import ExitStack


def log(msg):
  if msg[-1] != '\n':
    msg += '\n'
  sys.stderr.write(msg)
  sys.stderr.flush()


def load_trained_variable(model_path, name):
  data = tf.train.load_variable(model_path, name)
  return data


def sharpen(p_model_y, temp):
  p_target = tf.pow(p_model_y, 1. / temp)
  p_target /= tf.reduce_sum(p_target, axis=-1, keep_dims=True)
  return p_target


def make_parallel(fn, num_gpus, **kwargs):
  in_splits = {}
  for k, v in kwargs.items():
    in_splits[k] = tf.split(v, num_gpus)

  out_splits = []

  for i in range(num_gpus):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        out_splits.append(fn(**{k: v[i] for k, v in in_splits.items()}))

  return tf.concat(out_splits, axis=0)


def aggregate_gradients(tower_grads, aggregate_fn=tf.reduce_mean):
  aggregate_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = aggregate_fn(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    aggregate_grads.append(grad_and_var)
  return aggregate_grads


def rigid_op_sequence(op_lambdas):
  with ExitStack() as stack:
    for op_func in op_lambdas:
      op = op_func()
      context = tf.control_dependencies([op])
      stack.enter_context(context)
  return op
