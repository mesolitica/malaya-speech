import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from glob import glob
from itertools import cycle
import random
import tensorflow as tf
from malaya_speech.train.model import fastspeech, fastvc
from math import ceil
from functools import partial
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
import malaya_speech.train as train
import malaya_speech
import random

speaker_model = malaya_speech.speaker_vector.deep_model('vggvox-v2')

mels = glob('output-universal/mels/*.npy')
file_cycle = cycle(mels)


def generate(hop_size=256):
    while True:
        f = next(file_cycle)
        mel = np.load(f)
        audio = np.load(f.replace('mels', 'audios'))

        batch_max_steps = random.randint(16384, 110250)
        batch_max_frames = batch_max_steps // hop_size
        if len(audio) < len(mel) * hop_size:
            audio = np.pad(audio, [[0, len(mel) * hop_size - len(audio)]])

        if len(mel) > batch_max_frames:
            interval_start = 0
            interval_end = len(mel) - batch_max_frames
            start_frame = random.randint(interval_start, interval_end)
            start_step = start_frame * hop_size
            audio = audio[start_step: start_step + batch_max_steps]
            mel = mel[start_frame: start_frame + batch_max_frames, :]
        else:
            audio = np.pad(audio, [[0, batch_max_steps - len(audio)]])
            mel = np.pad(mel, [[0, batch_max_frames - len(mel)], [0, 0]])

        v = speaker_model([audio])

        yield {'mel': mel, 'mel_length': [len(mel)], 'audio': audio, 'v': v[0]}


def get_dataset(batch_size=12):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'mel_length': tf.int32,
                'audio': tf.float32,
                'v': tf.float32,
            },
            output_shapes={
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'audio': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
        )
        dataset = dataset.shuffle(batch_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'audio': tf.TensorShape([None]),
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
            padding_values={
                'audio': tf.constant(0, dtype=tf.float32),
                'mel': tf.constant(0, dtype=tf.float32),
                'mel_length': tf.constant(0, dtype=tf.int32),
                'v': tf.constant(0, dtype=tf.float32),
            },
        )
        return dataset

    return get


total_steps = 300000


def model_fn(features, labels, mode, params):
    vectors = features['v'] * 30 - 3.5
    mels = features['mel']
    mels_len = features['mel_length'][:, 0]
    dim_neck = 64
    config = malaya_speech.config.fastspeech_config
    config['encoder_hidden_size'] = 512 + 80
    config['decoder_hidden_size'] = 512 + dim_neck
    config = fastspeech.Config(vocab_size=1, **config)
    model = fastvc.model.Model(dim_neck, config)
    encoder_outputs, mel_before, mel_after, codes = model(
        mels, vectors, vectors, mels_len
    )
    codes_ = model.call_second(mel_after, vectors, mels_len)
    loss_f = tf.losses.absolute_difference
    max_length = tf.cast(tf.reduce_max(mels_len), tf.int32)
    mask = tf.sequence_mask(
        lengths=mels_len, maxlen=max_length, dtype=tf.float32
    )
    mask = tf.expand_dims(mask, axis=-1)
    mel_loss_before = loss_f(
        labels=mels, predictions=mel_before, weights=mask
    )
    mel_loss_after = loss_f(
        labels=mels, predictions=mel_after, weights=mask
    )
    g_loss_cd = tf.losses.absolute_difference(codes, codes_)
    loss = mel_loss_before + mel_loss_after + g_loss_cd

    tf.identity(loss, 'total_loss')
    tf.identity(mel_loss_before, 'mel_loss_before')
    tf.identity(mel_loss_after, 'mel_loss_after')
    tf.identity(g_loss_cd, 'g_loss_cd')

    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('mel_loss_before', mel_loss_before)
    tf.summary.scalar('mel_loss_after', mel_loss_after)
    tf.summary.scalar('g_loss_cd', g_loss_cd)

    global_step = tf.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:

        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr=0.001,
            num_train_steps=total_steps,
            num_warmup_steps=int(0.1 * total_steps),
            end_learning_rate=0.00005,
            weight_decay_rate=0.001,
            beta_1=0.9,
            beta_2=0.98,
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
        ['total_loss', 'mel_loss_before', 'mel_loss_after', 'g_loss_cd'],
        every_n_iter=1,
    )
]
train_dataset = get_dataset()

save_directory = 'fastvc-64'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=2000,
    max_steps=total_steps,
    train_hooks=train_hooks,
    eval_step=0,
)
