import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.train.model.quartznet as quartznet
import malaya_speech.train.model.ctc as ctc
import malaya_speech.train as train
import numpy as np
import random
from glob import glob
import json

with open('malaya-speech-sst-vocab.json') as fopen:
    unique_vocab = json.load(fopen)

parameters = {
    'optimizer_params': {
        'beta1': 0.95,
        'beta2': 0.5,
        'epsilon': 1e-08,
        'weight_decay': 0.001,
        'grad_averaging': False,
    },
    'lr_policy_params': {
        'learning_rate': 0.01,
        'min_lr': 0.0,
        'warmup_steps': 1000,
        'decay_steps': 100_000,
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


noises = glob('../noise-44k/noise/*.wav') + glob('../noise-44k/clean-wav/*.wav')
basses = glob('HHDS/Sources/**/*bass.wav', recursive = True)
drums = glob('HHDS/Sources/**/*drums.wav', recursive = True)
others = glob('HHDS/Sources/**/*other.wav', recursive = True)
noises = noises + basses + drums + others
random.shuffle(noises)


def read_wav(f):
    return malaya_speech.load(f, sr = 16000)


def random_amplitude_threshold(sample, low = 1, high = 2, threshold = 0.4):
    y_aug = sample.copy()
    y_aug = y_aug / (np.max(np.abs(y_aug)) + 1e-9)
    dyn_change = np.random.uniform(low = low, high = high)
    y_aug[np.abs(y_aug) >= threshold] = (
        y_aug[np.abs(y_aug) >= threshold] * dyn_change
    )
    return np.clip(y_aug, -1, 1)


def calc(signal, seed, add_uniform = False):
    random.seed(seed)

    choice = random.randint(0, 9)
    if choice == 0:

        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain = random.randint(10, 30),
            reverberance = random.randint(0, 30),
            hf_damping = 10,
            room_scale = random.randint(0, 30),
            negate = 1,
        )
    if choice == 1:
        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain = random.randint(10, 40),
            reverberance = random.randint(0, 30),
            hf_damping = 10,
            room_scale = random.randint(0, 30),
            negate = 0,
        )
    if choice == 2:
        x = augmentation.sox_augment_low(
            signal,
            min_bass_gain = random.randint(1, 20),
            reverberance = random.randint(0, 30),
            hf_damping = 10,
            room_scale = random.randint(0, 30),
            negate = random.randint(0, 1),
        )
    if choice == 3:
        x = augmentation.sox_augment_combine(
            signal,
            min_bass_gain_high = random.randint(10, 40),
            min_bass_gain_low = random.randint(1, 20),
            reverberance = random.randint(0, 30),
            hf_damping = 10,
            room_scale = random.randint(0, 30),
        )
    if choice == 4:
        x = augmentation.sox_reverb(
            signal,
            reverberance = random.randint(1, 20),
            hf_damping = 10,
            room_scale = random.randint(10, 30),
        )
    if choice == 5:
        x = random_amplitude_threshold(
            signal, threshold = random.uniform(0.35, 0.8)
        )

    if choice > 5:
        x = signal

    if choice != 5 and random.gauss(0.5, 0.14) > 0.6:
        x = random_amplitude_threshold(
            x, low = 1.0, high = 2.0, threshold = random.uniform(0.7, 0.9)
        )

    if random.gauss(0.5, 0.14) > 0.6 and add_uniform:
        x = augmentation.add_uniform_noise(
            x, power = random.uniform(0.005, 0.015)
        )

    return x


def signal_augmentation(wav):
    seed = random.randint(0, 100_000_000)
    wav = calc(wav, seed)
    if random.gauss(0.5, 0.14) > 0.6:
        n, _ = malaya_speech.load(random.choice(noises), sr = 16000)
        n = calc(n, seed, True)
        combined = augmentation.add_noise(
            wav, n, factor = random.uniform(0.05, 0.2)
        )
    else:
        combined = wav
    return combined.astype('float32')


def mel_augmentation(features):
    features = mask_augmentation.mask_frequency(features)
    return mask_augmentation.mask_time(features)


# def preprocess_inputs(example):
#     w = tf.compat.v1.numpy_function(
#         signal_augmentation, [example['waveforms']], tf.float32
#     )
#     w = tf.reshape(w, (1, -1))
#     s = featurizer.vectorize(w[0])
#     s = featurizer.vectorize(example['waveforms'])
#     s = tf.reshape(s, (-1, n_mels))
#     s = tf.compat.v1.numpy_function(mel_augmentation, [s], tf.float32)
#     mel_fbanks = tf.reshape(s, (-1, n_mels))
#     length = tf.cast(tf.shape(mel_fbanks)[0], tf.int32)
#     length = tf.expand_dims(length, 0)
#     example['waveforms'] = w[0]
#     example['inputs'] = mel_fbanks
#     example['inputs_length'] = length

#     return example


def preprocess_inputs(example):
    s = featurizer.vectorize(example['waveforms'])
    s = tf.reshape(s, (-1, n_mels))
    s = malaya_speech.augmentation.spectrogram.tf_mask_frequency(s, F = 20)
    s = malaya_speech.augmentation.spectrogram.tf_mask_time(s, T = 80)
    mel_fbanks = tf.reshape(s, (-1, n_mels))
    length = tf.cast(tf.shape(mel_fbanks)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['waveforms'] = w[0]
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

    model = quartznet.Model(
        features['inputs'], features['inputs_length'][:, 0], mode = 'train'
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

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.optimize_loss(
            loss,
            train.optimizer.NovoGrad,
            parameters['optimizer_params'],
            learning_rate_scheduler,
            summaries = parameters.get('summaries', None),
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
dev_dataset = get_dataset('../speech-bahasa/bahasa-asr/data/bahasa-asr-dev-*')

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = 'asr-quartznet',
    num_gpus = 3,
    log_step = 1,
    save_checkpoint_step = 2000,
    max_steps = parameters['lr_policy_params']['decay_steps'],
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
