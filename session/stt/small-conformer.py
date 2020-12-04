import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.train.model.conformer as conformer
import malaya_speech.train.model.transducer as transducer
import malaya_speech.config
import malaya_speech.train as train
import numpy as np
import random
from glob import glob
import json

subwords = malaya_speech.subword.load('malaya-speech.tokenizer')
with open('malaya-speech-sst-vocab.json') as fopen:
    unique_vocab = json.load(fopen) + ['{', '}', '[']

config = malaya_speech.config.conformer_small_encoder_config

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 10e-9},
    'lr_policy_params': {
        'max_lr': 0.05 / np.sqrt(config['dmodel']),
        'warmup_steps': 10000,
    },
}


def transformer_schedule(step, d_model, warmup_steps = 4000, max_lr = None):
    arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
    arg2 = step * (warmup_steps ** -1.5)
    arg1 = tf.cast(arg1, tf.float32)
    arg2 = tf.cast(arg2, tf.float32)
    lr = tf.math.rsqrt(tf.cast(d_model, tf.float32)) * tf.math.minimum(
        arg1, arg2
    )
    if max_lr is not None:
        max_lr = tf.cast(max_lr, tf.float32)
        return tf.math.minimum(max_lr, lr)
    return lr


def learning_rate_scheduler(global_step):

    return transformer_schedule(
        tf.cast(global_step, tf.float32),
        config['dmodel'],
        **parameters['lr_policy_params'],
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


def char_to_subwords(features):
    t = malaya_speech.char.decode(features, lookup = unique_vocab).replace(
        '<PAD>', ''
    )
    t = malaya_speech.subword.encode(subwords, t, add_blank = True)
    return np.array(t)


def preprocess_inputs(example):
    s = featurizer.vectorize(example['waveforms'])
    s = tf.reshape(s, (-1, n_mels))
    s = tf.compat.v1.numpy_function(mel_augmentation, [s], tf.float32)
    mel_fbanks = tf.reshape(s, (-1, n_mels))
    length = tf.cast(tf.shape(mel_fbanks)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['inputs'] = mel_fbanks
    example['inputs_length'] = length

    s = tf.compat.v1.numpy_function(
        char_to_subwords, [example['targets']], tf.int64
    )
    s = tf.reshape(s, (1, -1))[0]

    example['targets'] = s
    example['targets_length'] = tf.expand_dims(
        tf.cast(tf.shape(s)[0], tf.int32), 0
    )
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
        if k not in [
            'waveforms',
            'inputs',
            'inputs_length',
            'targets',
            'targets_length',
        ]:
            features.pop(k, None)

    return features


def get_dataset(
    path,
    batch_size = 16,
    shuffle_size = 16,
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
                'targets_length': tf.TensorShape([None]),
            },
            padding_values = {
                'waveforms': tf.constant(0, dtype = tf.float32),
                'inputs': tf.constant(0, dtype = tf.float32),
                'inputs_length': tf.constant(0, dtype = tf.int32),
                'targets': tf.constant(0, dtype = tf.int64),
                'targets_length': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    conformer_model = conformer.Model(**config)
    decoder_config = malaya_speech.config.conformer_small_decoder_config
    transducer_model = transducer.rnn.Model(
        conformer_model, vocabulary_size = subwords.vocab_size, **decoder_config
    )
    v = tf.expand_dims(features['inputs'], -1)
    z = tf.zeros((tf.shape(features['targets'])[0], 1), dtype = tf.int64)
    c = tf.concat([z, features['targets']], axis = 1)

    logits = transducer_model([v, c], training = True)

    cost = transducer.loss.rnnt_loss(
        logits = tf.nn.log_softmax(logits),
        labels = features['targets'],
        label_length = features['targets_length'][:, 0],
        logit_length = features['inputs_length'][:, 0]
        // conformer_model.conv_subsampling.time_reduction_factor,
    )
    mean_error = tf.reduce_mean(cost)

    loss = mean_error

    tf.identity(loss, 'train_loss')

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
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


train_hooks = [tf.train.LoggingTensorHook(['train_loss'], every_n_iter = 1)]
train_dataset = get_dataset(
    '../speech-bahasa/bahasa-asr/data/bahasa-asr-train-*'
)
dev_dataset = get_dataset(
    '../speech-bahasa/bahasa-asr-test/data/bahasa-asr-dev-*'
)

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = 'asr-small-conformer-transducer',
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 5000,
    max_steps = 500_000,
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
