import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle
import tensorflow as tf
import malaya_speech
import malaya_speech.train
import malaya_speech.config
import malaya_speech.train as train
from malaya_speech.train.model import fastspeech2
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
from functools import partial
import json
import re

with open('mels-male.json') as fopen:
    files = json.load(fopen)


def norm_mean_std(x, mean, std):
    zero_idxs = np.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x


def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    x_char = np.zeros((durs.shape[0],), dtype = np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0

    return x_char.astype(np.float32)


def get_alignment(f):
    f = f"tacotron2-male-alignment/{f.split('/')[-1]}"
    if os.path.exists(f):
        return np.load(f)
    else:
        return None


f0_stat = np.load('../speech-bahasa/male-stats/stats_f0.npy')
energy_stat = np.load('../speech-bahasa/male-stats/stats_energy.npy')

reduction_factor = 1
maxlen = 904
minlen = 32
pad_to = 8
data_min = 1e-2

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

MALAYA_SPEECH_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)

parameters = {
    'optimizer_params': {},
    'lr_policy_params': {
        'learning_rate': 1e-3,
        'decay_steps': 20000,
        'decay_rate': 0.1,
        'use_staircase_decay': False,
        'begin_decay_at': 45000,
        'min_lr': 1e-5,
    },
    'max_grad_norm': 1.0,
}


def exp_decay(
    global_step,
    learning_rate,
    decay_steps,
    decay_rate,
    use_staircase_decay,
    begin_decay_at = 0,
    min_lr = 0.0,
):
    new_lr = tf.cond(
        global_step < begin_decay_at,
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(
            learning_rate = learning_rate,
            global_step = global_step - begin_decay_at,
            decay_steps = decay_steps,
            decay_rate = decay_rate,
            staircase = use_staircase_decay,
        ),
        name = 'learning_rate',
    )
    final_lr = tf.maximum(min_lr, new_lr)
    return final_lr


def learning_rate_scheduler(global_step):
    return exp_decay(global_step, **parameters['lr_policy_params'])


def generate(files):
    file_cycle = cycle(files)
    while True:
        f = next(file_cycle).decode()
        mel = np.load(f)
        mel_length = len(mel)
        if mel_length > maxlen or mel_length < minlen:
            continue

        alignment = get_alignment(f)
        if alignment is None:
            continue

        stop_token_target = np.zeros([len(mel)], dtype = np.float32)

        text_ids = np.load(f.replace('mels', 'text_ids'), allow_pickle = True)[
            0
        ]
        text_input = np.array(
            [
                MALAYA_SPEECH_SYMBOLS.index(c)
                for c in text_ids
                if c in MALAYA_SPEECH_SYMBOLS
            ]
        )
        num_pad = pad_to - ((len(text_input) + 2) % pad_to)
        text_input = np.pad(
            text_input, ((1, 1)), 'constant', constant_values = ((1, 2))
        )
        text_input = np.pad(
            text_input, ((0, num_pad)), 'constant', constant_values = 0
        )
        num_pad = pad_to - ((len(mel) + 1) % pad_to) + 1
        pad_value_mel = np.log(data_min)
        mel = np.pad(
            mel,
            ((0, num_pad), (0, 0)),
            'constant',
            constant_values = pad_value_mel,
        )
        stop_token_target = np.pad(
            stop_token_target, ((0, num_pad)), 'constant', constant_values = 1
        )
        len_mel = [len(mel)]
        len_text_ids = [len(text_input)]

        f0 = np.load(f.replace('mels', 'f0s'))
        f0 = norm_mean_std(f0, f0_stat[0], f0_stat[1])
        f0 = average_by_duration(f0, alignment)
        len_f0 = [len(f0)]

        energy = np.load(f.replace('mels', 'energies'))
        energy = norm_mean_std(energy, energy_stat[0], energy_stat[1])
        energy = average_by_duration(energy, alignment)
        len_energy = [len(energy)]

        yield {
            'mel': mel,
            'text_ids': text_input,
            'len_mel': len_mel,
            'len_text_ids': len_text_ids,
            'stop_token_target': stop_token_target,
            'f0': f0,
            'len_f0': len_f0,
            'energy': energy,
            'len_energy': len_energy,
            'f': [f],
            'alignment': alignment,
        }


def get_dataset(files, batch_size = 32, shuffle_size = 32, thread_count = 24):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'text_ids': tf.int32,
                'len_mel': tf.int32,
                'len_text_ids': tf.int32,
                'stop_token_target': tf.float32,
                'f0': tf.float32,
                'len_f0': tf.int32,
                'energy': tf.float32,
                'len_energy': tf.int32,
                'f': tf.string,
                'alignment': tf.int32,
            },
            output_shapes = {
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'stop_token_target': tf.TensorShape([None]),
                'f0': tf.TensorShape([None]),
                'len_f0': tf.TensorShape([1]),
                'energy': tf.TensorShape([None]),
                'len_energy': tf.TensorShape([1]),
                'f': tf.TensorShape([1]),
                'alignment': tf.TensorShape([None]),
            },
            args = (files,),
        )
        dataset = dataset.padded_batch(
            shuffle_size,
            padded_shapes = {
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'stop_token_target': tf.TensorShape([None]),
                'f0': tf.TensorShape([None]),
                'len_f0': tf.TensorShape([1]),
                'energy': tf.TensorShape([None]),
                'len_energy': tf.TensorShape([1]),
                'f': tf.TensorShape([1]),
                'alignment': tf.TensorShape([None]),
            },
            padding_values = {
                'mel': tf.constant(0, dtype = tf.float32),
                'text_ids': tf.constant(0, dtype = tf.int32),
                'len_mel': tf.constant(0, dtype = tf.int32),
                'len_text_ids': tf.constant(0, dtype = tf.int32),
                'stop_token_target': tf.constant(0, dtype = tf.float32),
                'f0': tf.constant(0, dtype = tf.float32),
                'len_f0': tf.constant(0, dtype = tf.int32),
                'energy': tf.constant(0, dtype = tf.float32),
                'len_energy': tf.constant(0, dtype = tf.int32),
                'f': tf.constant('', dtype = tf.string),
                'alignment': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    input_ids = features['text_ids']
    input_lengths = features['len_text_ids'][:, 0]
    mel_outputs = features['mel']
    mel_lengths = features['len_mel'][:, 0]
    energies = features['energy']
    energies_lengths = features['len_energy'][:, 0]
    f0s = features['f0']
    f0s_lengths = features['len_f0'][:, 0]
    batch_size = tf.shape(f0s)[0]
    alignment = features['alignment']

    config = malaya_speech.config.fastspeech2_config_v2
    config = fastspeech2.Config(
        vocab_size = len(MALAYA_SPEECH_SYMBOLS), **config
    )
    model = fastspeech2.Model(config)

    mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs = model(
        input_ids, alignment, f0s, energies, training = True
    )
    loss_f = tf.losses.mean_squared_error
    log_duration = tf.math.log(tf.cast(tf.math.add(alignment, 1), tf.float32))
    duration_loss = loss_f(log_duration, duration_outputs)
    max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
    mask = tf.sequence_mask(
        lengths = mel_lengths, maxlen = max_length, dtype = tf.float32
    )
    mask = tf.expand_dims(mask, axis = -1)

    mse_mel = partial(loss_f, weights = mask)
    mel_loss_before = calculate_3d_loss(mel_outputs, mel_before, mse_mel)
    mel_loss_after = calculate_3d_loss(mel_outputs, mel_after, mse_mel)

    max_length = tf.cast(tf.reduce_max(energies_lengths), tf.int32)
    mask = tf.sequence_mask(
        lengths = energies_lengths, maxlen = max_length, dtype = tf.float32
    )
    mse_energies = partial(loss_f, weights = mask)
    energies_loss = calculate_2d_loss(energies, energy_outputs, mse_energies)
    max_length = tf.cast(tf.reduce_max(f0s_lengths), tf.int32)
    mask = tf.sequence_mask(
        lengths = f0s_lengths, maxlen = max_length, dtype = tf.float32
    )
    mse_f0s = partial(loss_f, weights = mask)
    f0s_loss = calculate_2d_loss(f0s, f0_outputs, mse_f0s)
    loss = (
        duration_loss
        + mel_loss_before
        + mel_loss_after
        + energies_loss
        + f0s_loss
    )

    tf.identity(loss, 'loss')
    tf.identity(duration_loss, name = 'duration_loss')
    tf.identity(mel_loss_before, name = 'mel_loss_before')
    tf.identity(mel_loss_after, name = 'mel_loss_after')
    tf.identity(energies_loss, name = 'energies_loss')
    tf.identity(f0s_loss, name = 'f0s_loss')

    tf.summary.scalar('duration_loss', duration_loss)
    tf.summary.scalar('mel_loss_before', mel_loss_before)
    tf.summary.scalar('mel_loss_after', mel_loss_after)
    tf.summary.scalar('energies_loss', energies_loss)
    tf.summary.scalar('f0s_loss', f0s_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.optimize_loss(
            loss,
            tf.train.AdamOptimizer,
            parameters['optimizer_params'],
            learning_rate_scheduler,
            summaries = ['learning_rate'],
            larc_params = parameters.get('larc_params', None),
            loss_scaling = parameters.get('loss_scaling', 1.0),
            loss_scaling_params = parameters.get('loss_scaling_params', None),
            clip_gradients = parameters.get('max_grad_norm', None),
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        [
            'loss',
            'duration_loss',
            'mel_loss_before',
            'mel_loss_after',
            'energies_loss',
            'f0s_loss',
        ],
        every_n_iter = 1,
    )
]

train_dataset = get_dataset(files['train'])
dev_dataset = get_dataset(files['test'])

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = 'fastspeech2-male',
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 2000,
    max_steps = 100_000,
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
