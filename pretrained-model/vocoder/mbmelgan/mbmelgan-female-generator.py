import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle

mels = glob('../speech-bahasa/output-female-v2/mels/*.npy')
file_cycle = cycle(mels)
f = next(file_cycle)

import random


def generate(batch_max_steps=8192, hop_size=256):
    while True:
        f = next(file_cycle)
        mel = np.load(f)
        audio = np.load(f.replace('mels', 'audios'))

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

        yield {'mel': mel, 'audio': audio}


dataset = tf.data.Dataset.from_generator(
    generate,
    {'mel': tf.float32, 'audio': tf.float32},
    output_shapes={
        'mel': tf.TensorShape([None, 80]),
        'audio': tf.TensorShape([None]),
    },
)
dataset = dataset.shuffle(32)
dataset = dataset.padded_batch(
    32,
    padded_shapes={
        'audio': tf.TensorShape([None]),
        'mel': tf.TensorShape([None, 80]),
    },
    padding_values={
        'audio': tf.constant(0, dtype=tf.float32),
        'mel': tf.constant(0, dtype=tf.float32),
    },
)

features = dataset.make_one_shot_iterator().get_next()
features

import malaya_speech
import malaya_speech.train
from malaya_speech.train.model import melgan, mb_melgan
from malaya_speech.train.model import stft
import malaya_speech.config
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss

mb_melgan_config = malaya_speech.config.mb_melgan_config
generator = melgan.Generator(
    mb_melgan.GeneratorConfig(**mb_melgan_config['melgan_generator_params']),
    name='mb_melgan-generator',
)
pqmf = mb_melgan.PQMF(
    mb_melgan.GeneratorConfig(**mb_melgan_config['melgan_generator_params']),
    dtype=tf.float32,
    name='pqmf',
)

sub_band_stft_loss = stft.loss.MultiResolutionSTFT(
    **mb_melgan_config['subband_stft_loss_params']
)

full_band_stft_loss = stft.loss.MultiResolutionSTFT(
    **mb_melgan_config['stft_loss_params']
)

y_mb_hat = generator(features['mel'], training=True)
audios = features['audio']
y_hat = pqmf.synthesis(y_mb_hat)

y_mb = pqmf.analysis(tf.expand_dims(audios, -1))
y_mb = tf.transpose(y_mb, (0, 2, 1))
y_mb = tf.reshape(y_mb, (-1, tf.shape(y_mb)[-1]))

y_mb_hat = tf.transpose(y_mb_hat, (0, 2, 1))
y_mb_hat = tf.reshape(y_mb_hat, (-1, tf.shape(y_mb_hat)[-1]))
sub_sc_loss, sub_mag_loss = calculate_2d_loss(
    y_mb, y_mb_hat, sub_band_stft_loss
)

sub_sc_loss = tf.reduce_mean(tf.reshape(sub_sc_loss, [-1, pqmf.subbands]), -1)
sub_mag_loss = tf.reduce_mean(tf.reshape(sub_mag_loss, [-1, pqmf.subbands]), -1)
full_sc_loss, full_mag_loss = calculate_2d_loss(
    audios, tf.squeeze(y_hat, -1), full_band_stft_loss
)

generator_loss = 0.5 * (sub_sc_loss + sub_mag_loss) + 0.5 * (
    full_sc_loss + full_mag_loss
)
generator_loss = tf.reduce_mean(generator_loss)
g_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9).minimize(
    generator_loss
)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

checkpoint = 10000
epoch = 200_000
path = 'mbmelgan-female'

ckpt_path = tf.train.latest_checkpoint(path)
if ckpt_path:
    saver.restore(sess, ckpt_path)
    print(f'restoring checkpoint from {ckpt_path}')

for i in range(0, epoch):
    g_loss, _ = sess.run([generator_loss, g_optimizer])

    if i % checkpoint == 0:
        saver.save(sess, f'{path}/model.ckpt', global_step=i)

    print(i, g_loss)

saver.save(sess, f'{path}/model.ckpt', global_step=epoch)
