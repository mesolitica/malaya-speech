import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/husein/t5/prepare/mesolitica-tpu.json'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.config
import malaya_speech.train as train
from malaya_speech.train.model import hubert, ctc
from malaya_speech.train.model.conformer.model import Model as ConformerModel
import json
import random
import numpy as np
import string

sr = 16000
maxlen = 16
minlen = 1
minlen_text = 1

unique_vocab = [''] + list(string.ascii_lowercase + string.digits) + [' ']


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


def get_dataset(files, batch_size=4, shuffle_size=32, num_cpu_threads=6,
                thread_count=24, is_training=True):
    def get():
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(files))
            cycle_length = min(num_cpu_threads, len(files))
            d = d.interleave(
                tf.data.TFRecordDataset,
                cycle_length=cycle_length,
                block_length=thread_count)
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(files)
            d = d.repeat()
        d = d.map(parse, num_parallel_calls=thread_count)
        d = d.filter(
            lambda x: tf.less(tf.shape(x['waveforms'])[0] / sr, maxlen)
        )
        d = d.filter(
            lambda x: tf.greater(tf.shape(x['waveforms'])[0] / sr, minlen)
        )
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

with open('3mixed-train-test-v2.json') as fopen:
    dataset = json.load(fopen)

train_dataset = get_dataset(dataset['train'], is_training=True)

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='hubert-conformer-base-3mixed-ctc',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=20000,
    max_steps=total_steps,
    eval_fn=None,
    train_hooks=train_hooks,
)
