import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.train.model.conformer as conformer
import malaya_speech.train.model.transducer as transducer
from malaya_speech.train.model import leaf
import malaya_speech.config
import malaya_speech.train as train
import json
import numpy as np
import random
from glob import glob

subwords = malaya_speech.subword.load('transducer.subword')
config = malaya_speech.config.conformer_small_encoder_config

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 10e-9},
    'lr_policy_params': {
        'warmup_steps': 40000,
        'max_lr': (0.05 / config['dmodel']),
    },
}

featurizer = malaya_speech.tf_featurization.STTFeaturizer(
    normalize_per_feature = True
)
n_mels = featurizer.num_feature_bins


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


def preprocess_inputs(example):
    example['waveforms_length'] = tf.shape(example['waveforms'])[0]
    example['waveforms_length'] = tf.expand_dims(example['waveforms_length'], 0)
    example['targets'] = tf.cast(example['targets'], tf.int32)
    example['targets_length'] = tf.expand_dims(
        tf.cast(tf.shape(example['targets'])[0], tf.int32), 0
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
            'waveforms_length',
            'targets',
            'targets_length',
        ]:
            features.pop(k, None)

    return features


def get_dataset(
    path,
    batch_size = 20,
    shuffle_size = 20,
    thread_count = 24,
    maxlen_feature = 16000 * 10,
):
    def get():
        files = glob(path)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.map(parse, num_parallel_calls = thread_count)
        dataset = dataset.filter(
            lambda x: tf.less_equal(tf.shape(x['waveforms'])[0], maxlen_feature)
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            padding_values = {
                'waveforms': tf.constant(0, dtype = tf.float32),
                'waveforms_length': tf.constant(0, dtype = tf.int32),
                'targets': tf.constant(0, dtype = tf.int32),
                'targets_length': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


def batch_leaf(X, X_len, leaf_featurizer, training = True):
    training = True
    batch_size = tf.shape(X)[0]
    leaf_f = leaf_featurizer(X, training = training)
    # pretty hacky
    padded_lens = tf.cast(
        tf.cast(X_len, tf.float32) // 159.734_042_553_191_5, tf.int32
    )
    features = tf.TensorArray(
        dtype = tf.float32,
        size = batch_size,
        dynamic_size = True,
        infer_shape = False,
    )
    maxlen = tf.shape(leaf_f)[1]

    init_state = (0, features)

    def condition(i, features):
        return i < batch_size

    def body(i, features):
        f = leaf_f[i, : padded_lens[i]]
        warped = malaya_speech.augmentation.spectrogram.tf_warp_time(f)
        masked = malaya_speech.augmentation.spectrogram.tf_mask_frequency(
            warped, F = 12
        )
        casted_len = tf.cast(
            tf.cast(tf.shape(masked)[0], tf.float32) * 0.05, tf.int32
        )
        f = malaya_speech.augmentation.spectrogram.tf_mask_time(
            masked, T = casted_len
        )
        f = tf.pad(f, [[0, maxlen - tf.shape(f)[0]], [0, 0]])
        return i + 1, features.write(i, f)

    _, padded_features = tf.while_loop(condition, body, init_state)
    padded_features = padded_features.stack()
    padded_features.set_shape((None, None, 80))
    padded_features = tf.expand_dims(padded_features, -1)
    return padded_features, padded_lens


def model_fn(features, labels, mode, params):
    X = features['waveforms']
    X_len = features['waveforms_length'][:, 0]
    leaf_featurizer = leaf.Model()
    v, logits_length = batch_leaf(X, X_len, leaf_featurizer, training = True)

    conformer_model = conformer.Model(
        kernel_regularizer = None, bias_regularizer = None, **config
    )
    decoder_config = malaya_speech.config.conformer_small_decoder_config
    transducer_model = transducer.rnn.Model(
        conformer_model, vocabulary_size = subwords.vocab_size, **decoder_config
    )

    targets_length = features['targets_length'][:, 0]
    z = tf.zeros((tf.shape(features['targets'])[0], 1), dtype = tf.int32)
    c = tf.concat([z, features['targets']], axis = 1)

    logits = transducer_model([v, c, targets_length + 1], training = True)

    cost = transducer.loss.rnnt_loss(
        logits = logits,
        labels = features['targets'],
        label_length = targets_length,
        logit_length = logits_length
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
train_dataset = get_dataset('bahasa-asr/data/bahasa-asr-train-*')
dev_dataset = get_dataset('bahasa-asr-test/data/bahasa-asr-dev-*')

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = 'asr-leaf-small-conformer-transducer',
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 5000,
    max_steps = 500_000,
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
