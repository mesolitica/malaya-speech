import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.config
import malaya_speech.train as train
from collections import defaultdict
from malaya_speech.train.model import hubert, ctc
from malaya_speech.train.model.conformer.model import Model as ConformerModel
import json
import random
import numpy as np
import string
import requests
import shutil
from multiprocessing import Pool

sr = 16000
maxlen = 16
minlen = 1
minlen_text = 1

unique_vocab = [''] + list(string.ascii_lowercase + string.digits) + [' ']

with open('huggingface-3mixed-train-test.json') as fopen:
    dataset = json.load(fopen)

with open('huggingface-khursani-malay.json') as fopen:
    khursani_dataset = json.load(fopen)

languages = defaultdict(list)

for f in dataset['train']:
    l = f.split('/')[-2]
    languages[l].append(f)

train_set = random.sample(languages['singlish'], 650) + \
    languages['malay'] + khursani_dataset

test_set = [
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/malay/2-25.tfrecord',
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/singlish/2-34.tfrecord'
]


def download_file_cloud(url, filename):
    while True:
        try:
            r = requests.get(url, stream=True)
            total_size = int(r.headers['content-length'])
            version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
            try:
                local_size = os.path.getsize(filename)
                if local_size == total_size:
                    print(f'{filename} local size matched with cloud size')
                    return version
            except Exception as e:
                print(e)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                for data in r.iter_content(chunk_size=1_048_576):
                    f.write(data)
            break
        except Exception as e:
            print(e)


def get_dataset(files, directory='tfrecord', overwrite_directory=True):
    os.makedirs(directory, exist_ok=True)
    if overwrite_directory:
        shutil.rmtree(directory)
    files_to_download = []
    for f in files:
        filename = os.path.join(directory, '-'.join(f.split('/')[-2:]))
        files_to_download.append((f, filename))

    pool = Pool(processes=len(files))
    pool.starmap(download_file_cloud, files_to_download)
    pool.close()
    pool.join()
    tfrecords = glob(f'{directory}/*.tfrecord')
    return tfrecords


def generate(files, directory, overwrite_directory):
    while True:
        random.shuffle(files)
        for i in range(0, len(files), batch_files):
            batch = files[i: i + batch_files]
            batch = [b.decode() if isinstance(b, bytes) else b for b in batch]
            directory = directory.decode() if isinstance(directory, bytes) else directory
            r = get_dataset(batch, directory=directory, overwrite_directory=overwrite_directory)
            print(r)
            yield r


def preprocess_inputs(example):
    length = tf.cast(tf.shape(example['waveforms'])[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['waveforms_length'] = length
    example['targets'] = tf.cast(example['targets'], tf.int32)
    example['targets_length'] = tf.cast(example['targets_length'], tf.int32)
    return example


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.compat.v1.VarLenFeature(tf.float32),
        'targets': tf.compat.v1.VarLenFeature(tf.int64),
        'targets_length': tf.compat.v1.VarLenFeature(tf.int64),
        'lang': tf.compat.v1.VarLenFeature(tf.int64),
    }
    features = tf.compat.v1.parse_single_example(
        serialized_example, features=data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    features = preprocess_inputs(features)

    keys = list(features.keys())
    for k in keys:
        if k not in ['waveforms', 'waveforms_length', 'targets', 'targets_length']:
            features.pop(k, None)

    return features


def get_datasets(
    files, directory, overwrite_directory,
    batch_size=2, shuffle_size=32, num_cpu_threads=4, thread_count=24
):
    def get():
        d = tf.data.Dataset.from_generator(
            generate, tf.string, output_shapes=tf.TensorShape([None]),
            args=(files, directory, overwrite_directory),
        )
        d = d.repeat(3)
        d = d.interleave(
            tf.data.TFRecordDataset,
            cycle_length=num_cpu_threads,
            block_length=thread_count)
        d = d.shuffle(buffer_size=100)
        d = d.map(parse, num_parallel_calls=num_cpu_threads)
        d = d.padded_batch(
            batch_size,
            padded_shapes={
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            padding_values={
                'waveforms': tf.constant(0, dtype=tf.float32),
                'waveforms_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int32),
                'targets_length': tf.constant(0, dtype=tf.int32),
            },
        )
        return d
    return get


class Encoder:
    def __init__(self, config):
        self.config = config
        self.encoder = ConformerModel(**self.config)

    def __call__(self, x, input_mask, training=True):
        return self.encoder(x, training=training)


total_steps = 2000000


def model_fn(features, labels, mode, params):
    config_conformer = malaya_speech.config.conformer_base_encoder_config
    config_conformer['subsampling']['type'] = 'none'
    config_conformer['dropout'] = 0.0
    encoder = Encoder(config_conformer)
    cfg = hubert.HuBERTConfig(
        extractor_mode='layer_norm',
        dropout=0.0,
        attention_dropout=0.0,
        encoder_layerdrop=0.0,
        dropout_input=0.0,
        dropout_features=0.0,
        final_dim=256,
    )
    model = hubert.Model(cfg, encoder, ['pad', 'eos', 'unk'] + [str(i) for i in range(100)])
    X = features['waveforms']
    X_len = features['waveforms_length'][:, 0]
    targets = features['targets']
    targets_int32 = tf.cast(targets, tf.int32)
    targets_length = features['targets_length'][:, 0]
    r = model(X, padding_mask=X_len, features_only=True, mask=False)
    logits = tf.layers.dense(r['x'], len(unique_vocab) + 1)
    seq_lens = tf.reduce_sum(
        tf.cast(tf.logical_not(r['padding_mask']), tf.int32), axis=1
    )
    mean_error, sum_error, sum_weight = ctc.loss.ctc_loss(
        logits, seq_lens, targets_int32, targets_length
    )
    loss = mean_error
    accuracy = ctc.metrics.ctc_sequence_accuracy(
        logits, seq_lens, targets_int32, targets_length,
    )

    tf.identity(loss, 'train_loss')
    tf.identity(accuracy, name='train_accuracy')

    tf.summary.scalar('train_accuracy', accuracy)

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    init_checkpoint = 'hubert-conformer-base-3mixed/model.ckpt-2000000'

    assignment_map, initialized_variable_names = train.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr=5e-5,
            num_train_steps=total_steps,
            num_warmup_steps=100000,
            end_learning_rate=0.0,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            clip_norm=1.0,
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter=1
    )
]

train_dataset = get_dataset(dataset['train'], is_training=True)

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='hubert-conformer-base-2mixed-ctc',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=20000,
    max_steps=total_steps,
    eval_fn=None,
    train_hooks=train_hooks,
)
