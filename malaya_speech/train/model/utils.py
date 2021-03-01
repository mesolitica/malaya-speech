# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_mask_between(start, end, maxlen):
    s = tf.math.logical_not(tf.sequence_mask(start, maxlen))
    e = tf.sequence_mask(end, maxlen)
    return tf.math.logical_and(s, e)


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype = numerator.dtype))
    return numerator / denominator
