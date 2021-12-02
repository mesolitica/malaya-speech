import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle
import tensorflow as tf
import malaya_speech
import malaya_speech.train
import malaya_speech.config
import malaya_speech.train as train
from malaya_speech.train.model import revsic_glowtts as glowtts
from functools import partial
import math
import json
import re

files = glob('/home/husein/speech-bahasa/output-female-singlish/mels/*.npy')

reduction_factor = 1
maxlen = 1008
minlen = 32
pad_to = 2
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

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 1e-9},
    'lr_policy_params': {
        'warmup_steps': 4000,
        'learning_rate': 1.0,
    },
}

config = glowtts.Config(mel=80, vocabs=len(MALAYA_SPEECH_SYMBOLS))


def noam_schedule(step, channels, learning_rate=1.0, warmup_steps=4000):
    return learning_rate * channels ** -0.5 * \
        tf.minimum(step ** -0.5, step * warmup_steps ** -1.5)


def learning_rate_scheduler(global_step):
    return noam_schedule(
        tf.cast(global_step, tf.float32),
        config.channels,
        **parameters['lr_policy_params'],
    )


total_steps = 100_000


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

    model = glowtts.Model(config)
    loss, losses, attn = model.compute_loss(text=input_ids,
                                            textlen=input_lengths,
                                            mel=mel_outputs,
                                            mellen=mel_lengths)

    l_mle = losses['nll']
    l_length = losses['durloss']

    tf.identity(loss, 'loss')
    tf.identity(l_mle, name='l_mle')
    tf.identity(l_length, name='l_length')

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('l_mle', l_mle)
    tf.summary.scalar('l_length', l_length)

    global_step = tf.train.get_or_create_global_step()
    lr = learning_rate_scheduler(global_step)

    tf.summary.scalar('learning_rate', lr)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-09)
        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(loss, tvars)
        gvs = [(g, v) for g, v in gvs if g is not None]
        grads, tvars = list(zip(*gvs))
        all_finite = tf.constant(True, dtype=tf.bool)
        (grads, _) = tf.clip_by_global_norm(
            grads,
            clip_norm=2.0,
            use_norm=tf.cond(
                all_finite, lambda: tf.global_norm(grads), lambda: tf.constant(1.0)
            ),
        )
        train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step
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
        [
            'loss',
            'l_mle',
            'l_length',
        ],
        every_n_iter=1,
    )
]

train_dataset = get_dataset(files)

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='glowtts-female-singlish',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=2500,
    max_steps=total_steps,
    train_hooks=train_hooks,
)
