# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
import numpy as np
import six
import os
import random

UNSHUFFLED_SUFFIX = '-unshuffled'


def read_records(filename):
    reader = tf.python_io.tf_record_iterator(filename)
    records = []
    for record in reader:
        records.append(record)
        if len(records) % 100000 == 0:
            tf.logging.info('read: %d', len(records))
    return records


def write_records(records, out_filename):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for count, record in enumerate(records):
        writer.write(record)
        if count > 0 and count % 100000 == 0:
            tf.logging.info('write: %d', count)
    writer.close()


def _shuffle_single(fname, extra_fn=None):
    """Shuffle a single file of records.
  Args:
    fname: a string
    extra_fn: an optional function from list of TFRecords to list of TFRecords
      to be called after shuffling.
  """
    records = read_records(fname)
    random.shuffle(records)
    if extra_fn is not None:
        records = extra_fn(records)
    out_fname = fname.replace(UNSHUFFLED_SUFFIX, '')
    write_records(records, out_fname)
    tf.gfile.Remove(fname)


def shuffle_dataset(filenames, extra_fn=None):
    """Shuffles the dataset.
  Args:
    filenames: a list of strings
    extra_fn: an optional function from list of records to list of records
      to be called after shuffling a file.
  """
    if outputs_exist(filenames):
        tf.logging.info('Skipping shuffle because output files exist')
        return
    tf.logging.info('Shuffling data...')
    for filename in filenames:
        _shuffle_single(filename, extra_fn=extra_fn)
    tf.logging.info('Data shuffled.')


def sharded_name(base_name, shard, total_shards):
    return '%s-%.5d-of-%.5d' % (base_name, shard, total_shards)


def shard_filepath(fname, num_shards):
    return [
        sharded_name(fname, shard, num_shards) for shard in range(num_shards)
    ]


def outputs_exist(filenames):
    for out_fname in filenames:
        out_fname = out_fname.replace(UNSHUFFLED_SUFFIX, '')
        if tf.gfile.Exists(out_fname):
            return out_fname


def _data_filenames(output_name, output_dir, num_shards):
    return [
        os.path.join(output_dir, fname)
        for fname in shard_filepath(output_name, num_shards)
    ]


def train_data_filenames(problem, output_dir, num_shards):
    return _data_filenames(problem + '-train', output_dir, num_shards)


def dev_data_filenames(problem, output_dir, num_shards):
    return _data_filenames(problem + '-dev', output_dir, num_shards)


def test_data_filenames(problem, output_dir, num_shards):
    return _data_filenames(problem + '-test', output_dir, num_shards)


def training_filepaths(file_basename, data_dir, num_shards, shuffled):
    if not shuffled:
        file_basename += UNSHUFFLED_SUFFIX
    return train_data_filenames(file_basename, data_dir, num_shards)


def dev_filepaths(file_basename, data_dir, num_shards, shuffled):
    if not shuffled:
        file_basename += UNSHUFFLED_SUFFIX
    return dev_data_filenames(file_basename, data_dir, num_shards)


def test_filepaths(file_basename, data_dir, num_shards, shuffled):
    if not shuffled:
        file_basename += UNSHUFFLED_SUFFIX
    return test_data_filenames(file_basename, data_dir, num_shards)


def to_example(dictionary):
    """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
    features = {}
    for (k, v) in six.iteritems(dictionary):
        if not v:
            raise ValueError('Empty generated field: %s' % str((k, v)))
        # Subtly in PY2 vs PY3, map is not scriptable in py3. As a result,
        # map objects will fail with TypeError, unless converted to a list.
        if six.PY3 and isinstance(v, map):
            v = list(v)
        if isinstance(v[0], six.integer_types) or np.issubdtype(
            type(v[0]), np.integer
        ):
            features[k] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=v)
            )
        elif isinstance(v[0], float):
            features[k] = tf.train.Feature(
                float_list=tf.train.FloatList(value=v)
            )
        elif isinstance(v[0], six.string_types):
            if not six.PY2:  # Convert in python 3.
                v = [bytes(x, 'utf-8') for x in v]
            features[k] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=v)
            )
        elif isinstance(v[0], bytes):
            features[k] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=v)
            )
        else:
            raise ValueError(
                'Value for %s is not a recognized type; v: %s type: %s'
                % (k, str(v[0]), str(type(v[0])))
            )
    return tf.train.Example(features=tf.train.Features(feature=features))


def generate_files(
    generator, output_filenames, max_cases=None, cycle_every_n=1
):
    """Generate cases from a generator and save as TFRecord files.
  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.
  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filenames: List of output file paths.
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.
    cycle_every_n: how many cases from the generator to take before
      switching to the next shard; by default set to 1, switch every case.
  """
    if outputs_exist(output_filenames):
        tf.logging.info(
            'Skipping generator because outputs files exists at {}'.format(
                output_filenames
            )
        )
        return
    tmp_filenames = [fname + '.incomplete' for fname in output_filenames]
    num_shards = len(output_filenames)
    if num_shards > 0:
        if '-train' in output_filenames[0]:
            tag = 'train'
        elif '-dev' in output_filenames[0]:
            tag = 'eval'
        else:
            tag = 'other'

    writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filenames]
    counter, shard = 0, 0
    for case in generator:
        if case is None:
            continue
        if counter % 100000 == 0:
            tf.logging.info('Generating case %d.' % counter)
        counter += 1
        if max_cases and counter > max_cases:
            break
        example = to_example(case)
        writers[shard].write(example.SerializeToString())
        if counter % cycle_every_n == 0:
            shard = (shard + 1) % num_shards

    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_filenames, output_filenames):
        tf.gfile.Rename(tmp_name, final_name)

    tf.logging.info('Generated %s Examples', counter)


def check_shard(shards):
    for shard in shards:
        if 'split' not in shard.keys() or 'shards' not in shard.keys():
            raise ValueError('a shard must got `split` and `shards` keys')

        if shard['split'] not in ['train', 'test', 'dev']:
            raise ValueError(
                '`split` must be an element of [`train`, `test`, `dev`]'
            )
