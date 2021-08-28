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
from malaya_speech.train.model import glowtts
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
from functools import partial
import math
import json
import re

files = glob('../speech-bahasa/output-haqkiem/mels/*.npy')

reduction_factor = 1
maxlen = 1008
minlen = 32
pad_to = 8
data_min = 1e-2

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_rejected = '\'():;"'

MALAYA_SPEECH_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)

total_steps = 120_000


def generate(files):
    file_cycle = cycle(files)
    while True:
        f = next(file_cycle).decode()
        mel = np.load(f)
        mel_length = len(mel)
        if mel_length > maxlen or mel_length < minlen:
            continue

        stop_token_target = np.zeros([len(mel)], dtype=np.float32)

        text_ids = np.load(f.replace('mels', 'text_ids'), allow_pickle=True)[
            0
        ]
        text_ids = ''.join(
            [
                c
                for c in text_ids
                if c in MALAYA_SPEECH_SYMBOLS and c not in _rejected
            ]
        )
        text_ids = re.sub(r'[ ]+', ' ', text_ids).strip()
        text_input = np.array(
            [MALAYA_SPEECH_SYMBOLS.index(c) for c in text_ids]
        )
        num_pad = pad_to - ((len(text_input) + 2) % pad_to)
        text_input = np.pad(
            text_input, ((1, 1)), 'constant', constant_values=((1, 2))
        )
        text_input = np.pad(
            text_input, ((0, num_pad)), 'constant', constant_values=0
        )
        num_pad = pad_to - ((len(mel) + 1) % pad_to) + 1
        pad_value_mel = np.log(data_min)
        mel = np.pad(
            mel,
            ((0, num_pad), (0, 0)),
            'constant',
            constant_values=pad_value_mel,
        )
        len_mel = [len(mel)]
        len_text_ids = [len(text_input)]

        yield {
            'mel': mel,
            'text_ids': text_input,
            'len_mel': len_mel,
            'len_text_ids': len_text_ids,
            'f': [f],
        }


def get_dataset(files, batch_size=32, shuffle_size=32, thread_count=24):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'text_ids': tf.int32,
                'len_mel': tf.int32,
                'len_text_ids': tf.int32,
                'f': tf.string,
            },
            output_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'f': tf.TensorShape([1]),
            },
            args=(files,),
        )
        dataset = dataset.padded_batch(
            shuffle_size,
            padded_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'f': tf.TensorShape([1]),
            },
            padding_values={
                'mel': tf.constant(0, dtype=tf.float32),
                'text_ids': tf.constant(0, dtype=tf.int32),
                'len_mel': tf.constant(0, dtype=tf.int32),
                'len_text_ids': tf.constant(0, dtype=tf.int32),
                'f': tf.constant('', dtype=tf.string),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    input_ids = features['text_ids']
    input_lengths = features['len_text_ids'][:, 0]
    mel_outputs = features['mel']
    mel_lengths = features['len_mel'][:, 0]
    batch_size = tf.shape(mel_outputs)[0]

    config = malaya_speech.config.fastspeech2_config
    config['encoder_hidden_size'] = 192
    config['encoder_num_hidden_layers'] = 6
    config['encoder_attention_head_size'] = 32
    config['encoder_intermediate_size'] = 768
    config = glowtts.Config(vocab_size=len(MALAYA_SPEECH_SYMBOLS), **config)
    config_glowtts = glowtts.Config_GlowTTS(malaya_speech.config.glowtts_config)

    model = glowtts.Model(config, config_glowtts)
    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        input_ids, y=mel_outputs, y_lengths=mel_lengths, training=True)

    def mle_loss(z, m, logs, logdet, mask):
        l = tf.reduce_sum(logs) + 0.5 * tf.reduce_sum(tf.math.exp(-2 * logs) * ((z - m)**2))
        l = l - tf.reduce_sum(logdet)
        l = l / tf.reduce_sum(tf.ones_like(z) * mask)
        l = l + 0.5 * math.log(2 * math.pi)
        return l

    l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)

    def duration_loss(logw, logw_, lengths):
        l = tf.reduce_sum((logw - logw_)**2) / tf.reduce_sum(tf.cast(lengths, tf.float32))
        return l

    l_length = duration_loss(logw, logw_, input_lengths)
    loss_g = l_mle + l_length

    mae = tf.losses.absolute_difference
    max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)

    mask = tf.sequence_mask(
        lengths=mel_lengths, maxlen=max_length, dtype=tf.float32
    )
    mask = tf.expand_dims(mask, axis=-1)
    mel_loss = mae(
        labels=mel_outputs, predictions=z, weights=mask
    )

    tf.identity(loss_g, 'loss_g')
    tf.identity(l_mle, name='l_mle')
    tf.identity(l_length, name='l_length')
    tf.identity(mel_loss, name='mel_loss')

    tf.summary.scalar('loss_g', loss_g)
    tf.summary.scalar('l_mle', l_mle)
    tf.summary.scalar('l_length', l_length)
    tf.summary.scalar('mel_loss', mel_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.adamw.create_optimizer(
            loss_g,
            init_lr=0.001,
            num_train_steps=total_steps,
            num_warmup_steps=int(0.02 * total_steps),
            end_learning_rate=0.00005,
            weight_decay_rate=0.001,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6,
            clip_norm=1.0,
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_g, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss_g
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        [
            'loss_g',
            'l_mle',
            'l_length',
            'mel_loss',
        ],
        every_n_iter=1,
    )
]

train_dataset = get_dataset(files)

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='glowtts-haqkiem',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=10000,
    max_steps=total_steps,
    eval_fn=None,
    train_hooks=train_hooks,
)
