import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.train.model.mini_jasper as jasper
import malaya_speech.train.model.ctc as ctc
import malaya_speech.train as train
import numpy as np
import random
from glob import glob
import json

with open('malaya-speech-sst-vocab.json') as fopen:
    unique_vocab = json.load(fopen) + ['{', '}', '[']

parameters = {
    'optimizer_params': {},
    'lr_policy_params': {
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'warmup_steps': 0,
        'decay_steps': 500_000,
    },
}


def learning_rate_scheduler(global_step):
    return train.schedule.cosine_decay(
        global_step, **parameters['lr_policy_params']
    )


featurizer = malaya_speech.tf_featurization.STTFeaturizer(
    normalize_per_feature = True
)
n_mels = featurizer.num_feature_bins


def mel_augmentation(features):

    features = mask_augmentation.mask_frequency(features, width_freq_mask = 15)
    features = mask_augmentation.mask_time(
        features, width_time_mask = int(features.shape[0] * 0.05)
    )
    return features


def preprocess_inputs(example):
    s = featurizer.vectorize(example['waveforms'])
    s = tf.reshape(s, (-1, n_mels))
    s = tf.compat.v1.numpy_function(mel_augmentation, [s], tf.float32)
    mel_fbanks = tf.reshape(s, (-1, n_mels))
    length = tf.cast(tf.shape(mel_fbanks)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['inputs'] = mel_fbanks
    example['inputs_length'] = length

    return example


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.VarLenFeature(tf.float32),
        'targets': tf.VarLenFeature(tf.int64),
    }
    features = tf.parse_single_example(
        serialized_example, features = data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    features = preprocess_inputs(features)

    keys = list(features.keys())
    for k in keys:
        if k not in ['waveforms', 'inputs', 'inputs_length', 'targets']:
            features.pop(k, None)

    return features


def get_dataset(
    path,
    batch_size = 32,
    shuffle_size = 32,
    thread_count = 24,
    maxlen_feature = 1800,
):
    def get():
        files = glob(path)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.map(parse, num_parallel_calls = thread_count)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'waveforms': tf.TensorShape([None]),
                'inputs': tf.TensorShape([None, n_mels]),
                'inputs_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
            },
            padding_values = {
                'waveforms': tf.constant(0, dtype = tf.float32),
                'inputs': tf.constant(0, dtype = tf.float32),
                'inputs_length': tf.constant(0, dtype = tf.int32),
                'targets': tf.constant(0, dtype = tf.int64),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):

    model = jasper.Model(
        features['inputs'], features['inputs_length'][:, 0], training = True
    )
    logits = tf.layers.dense(model.logits['outputs'], len(unique_vocab) + 1)
    seq_lens = model.logits['src_length']

    targets_int32 = tf.cast(features['targets'], tf.int32)

    mean_error, sum_error, sum_weight = ctc.loss.ctc_loss(
        logits, targets_int32, seq_lens
    )

    loss = mean_error
    accuracy = ctc.metrics.ctc_sequence_accuracy(
        logits, targets_int32, seq_lens
    )

    tf.identity(loss, 'train_loss')
    tf.identity(accuracy, name = 'train_accuracy')

    tf.summary.scalar('train_accuracy', accuracy)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.optimize_loss(
            loss,
            tf.train.AdamOptimizer,
            parameters['optimizer_params'],
            learning_rate_scheduler,
            summaries = ['learning_rate', 'loss_scale'],
            larc_params = parameters.get('larc_params', None),
            loss_scaling = parameters.get('loss_scaling', 1.0),
            loss_scaling_params = parameters.get('loss_scaling_params', None),
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = loss,
            eval_metric_ops = {
                'accuracy': ctc.metrics.ctc_sequence_accuracy_estimator(
                    logits, targets_int32, seq_lens
                )
            },
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter = 1
    )
]
train_dataset = get_dataset(
    '../speech-bahasa/bahasa-asr/data/bahasa-asr-train-*'
)
dev_dataset = get_dataset(
    '../speech-bahasa/bahasa-asr-test/data/bahasa-asr-dev-*'
)

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = 'asr-mini-jasper-ctc',
    num_gpus = 2,
    log_step = 1,
    save_checkpoint_step = 5000,
    max_steps = parameters['lr_policy_params']['decay_steps'],
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
