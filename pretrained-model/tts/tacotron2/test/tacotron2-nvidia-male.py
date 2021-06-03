import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle
import tensorflow as tf
import malaya_speech
import malaya_speech.train
from malaya_speech.train.model import tacotron2_nvidia as tacotron2
import malaya_speech.config
import numpy as np
import json
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
import malaya_speech.train as train

with open('mels-male.json') as fopen:
    files = json.load(fopen)

import random

pad_to = 8
maxlen = 950
minlen = 32

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
    begin_decay_at=0,
    min_lr=0.0,
):
    new_lr = tf.cond(
        global_step < begin_decay_at,
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=global_step - begin_decay_at,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=use_staircase_decay,
        ),
        name='learning_rate',
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

        text_ids = np.load(f.replace('mels', 'text_ids'), allow_pickle=True)[
            1
        ]
        stop_token_target = np.zeros(len(mel), dtype=tf.float32)
        stop_token_target[-1] = 1.0
        len_mel = [len(mel)]
        len_text_ids = [len(text_ids)]

        yield {
            'mel': mel,
            'text_ids': text_ids,
            'len_mel': len_mel,
            'len_text_ids': len_text_ids,
            'stop_token_target': stop_token_target,
        }


def parse(example):
    mel_len = example['len_mel'][0]
    input_len = example['len_text_ids'][0]
    g = tacotron2.generate_guided_attention(
        mel_len, input_len, reduction_factor=reduction_factor
    )
    example['g'] = g
    return example


def get_dataset(files, batch_size=32, shuffle_size=32, thread_count=24):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'text_ids': tf.int32,
                'len_mel': tf.int32,
                'len_text_ids': tf.int32,
                'stop_token_target': tf.float32,
            },
            output_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'stop_token_target': tf.TensorShape([None]),
            },
            args=(files,),
        )
        dataset = dataset.map(parse, num_parallel_calls=thread_count)
        dataset = dataset.shuffle(batch_size)
        dataset = dataset.padded_batch(
            shuffle_size,
            padded_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'g': tf.TensorShape([None, None]),
                'stop_token_target': tf.TensorShape([None]),
            },
            padding_values={
                'mel': tf.constant(0, dtype=tf.float32),
                'text_ids': tf.constant(0, dtype=tf.int32),
                'len_mel': tf.constant(0, dtype=tf.int32),
                'len_text_ids': tf.constant(0, dtype=tf.int32),
                'g': tf.constant(-1.0, dtype=tf.float32),
                'stop_token_target': tf.constant(0, dtype=tf.float32),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    input_ids = features['text_ids']
    input_lengths = features['len_text_ids'][:, 0]
    speaker_ids = tf.constant([0], dtype=tf.int32)
    mel_outputs = features['mel']
    mel_lengths = features['len_mel'][:, 0]
    guided = features['g']

    model = tacotron2.Model(
        [input_ids, input_lengths],
        [mel_outputs, mel_lengths],
        len(MALAYA_SPEECH_SYMBOLS),
    )

    r = model.decoder_logits['outputs']
    decoder_output, post_mel_outputs, alignment_histories, _, _, _ = r
    stop_token_predictions = model.decoder_logits['stop_token_prediction']
    stop_token_predictions = stop_token_predictions[:, :, 0]

    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mae = tf.keras.losses.MeanAbsoluteError()

    mel_loss_before = calculate_3d_loss(
        mel_outputs, decoder_output, loss_fn=mae
    )
    mel_loss_after = calculate_3d_loss(
        mel_outputs, post_mel_outputs, loss_fn=mae
    )
    max_mel_length = tf.reduce_max(mel_lengths)
    stop_gts = tf.expand_dims(
        tf.range(tf.reduce_max(max_mel_length), dtype=tf.int32), 0
    )
    stop_gts = tf.tile(stop_gts, [tf.shape(mel_lengths)[0], 1])
    stop_gts = tf.cast(
        tf.math.greater_equal(stop_gts, tf.expand_dims(mel_lengths, 1)),
        tf.float32,
    )
    stop_token_loss = calculate_2d_loss(
        stop_gts, stop_token_predictions, loss_fn=binary_crossentropy
    )
    attention_masks = tf.cast(tf.math.not_equal(guided, -1.0), tf.float32)
    loss_att = tf.reduce_sum(
        tf.abs(alignment_histories * guided) * attention_masks, axis=[1, 2]
    )
    loss_att /= tf.reduce_sum(attention_masks, axis=[1, 2])
    loss_att = tf.reduce_mean(loss_att)

    loss = stop_token_loss + mel_loss_before + mel_loss_after + loss_att

    tf.identity(loss, 'loss')
    tf.identity(stop_token_loss, name='stop_token_loss')
    tf.identity(mel_loss_before, name='mel_loss_before')
    tf.identity(mel_loss_after, name='mel_loss_after')
    tf.identity(loss_att, name='loss_att')

    tf.summary.scalar('stop_token_loss', stop_token_loss)
    tf.summary.scalar('mel_loss_before', mel_loss_before)
    tf.summary.scalar('mel_loss_after', mel_loss_after)
    tf.summary.scalar('loss_att', loss_att)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.optimize_loss(
            loss,
            tf.train.AdamOptimizer,
            parameters['optimizer_params'],
            learning_rate_scheduler,
            summaries=[
                'learning_rate',
                'variables',
                'gradients',
                'larc_summaries',
                'variable_norm',
                'gradient_norm',
                'global_gradient_norm',
            ],
            larc_params=parameters.get('larc_params', None),
            loss_scaling=parameters.get('loss_scaling', 1.0),
            loss_scaling_params=parameters.get('loss_scaling_params', None),
            clip_gradients=parameters.get('max_grad_norm', None),
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
            'stop_token_loss',
            'mel_loss_before',
            'mel_loss_after',
            'loss_att',
        ],
        every_n_iter=1,
    )
]

train_dataset = get_dataset(files['train'])
dev_dataset = get_dataset(files['test'])

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='tacotron2-nvidia-male',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=5000,
    max_steps=150000,
    eval_fn=dev_dataset,
    train_hooks=train_hooks,
)
