import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle
import random

npys = glob('universal-audio/*.npy')
random.shuffle(npys)
file_cycle = cycle(npys)
f = next(file_cycle)


def generate(batch_max_steps=8192, hop_size=256):
    while True:
        f = next(file_cycle)
        audio = np.load(f)
        mel = np.load(f.replace('-audio', '-mel'))

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
from malaya_speech.train.model import melgan, hifigan
from malaya_speech.train.model import stft
import malaya_speech.config
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss

hifigan_config = malaya_speech.config.hifigan_config_v2
hifigan_config['hifigan_generator_params']['filters'] = 512
generator = hifigan.MultiGenerator(
    hifigan.GeneratorConfig(**hifigan_config['hifigan_generator_params']),
    name='hifigan_generator',
)

stft_loss = stft.loss.MultiResolutionSTFT(**hifigan_config['stft_loss_params'])

y_hat = generator(features['mel'], training=True)
audios = features['audio']

sc_loss, mag_loss = calculate_2d_loss(audios, tf.squeeze(y_hat, -1), stft_loss)

sc_loss = tf.where(sc_loss >= 15.0, tf.zeros_like(sc_loss), sc_loss)
mag_loss = tf.where(mag_loss >= 15.0, tf.zeros_like(mag_loss), mag_loss)

generator_loss = 0.5 * (sc_loss + mag_loss)
generator_loss = tf.reduce_mean(generator_loss)

global_step = tf.train.get_or_create_global_step()

print(global_step)

g_boundaries = [100000, 200000, 300000, 400000, 500000, 600000, 700000]
g_values = [0.000125, 0.000125, 0.0000625, 0.0000625, 0.0000625, 0.00003125, 0.000015625, 0.000001]

piece_wise = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    g_boundaries, g_values
)
lr = piece_wise(global_step)
g_optimizer = tf.train.AdamOptimizer(lr).minimize(
    generator_loss, global_step=global_step
)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

checkpoint = 10000
epoch = 50000
path = 'hifigan-512'

ckpt_path = tf.train.latest_checkpoint(path)
if ckpt_path:
    saver.restore(sess, ckpt_path)
    print(f'restoring checkpoint from {ckpt_path}')

for i in range(0, epoch):
    g_loss, _, step = sess.run([generator_loss, g_optimizer, global_step])

    if step % checkpoint == 0:
        saver.save(sess, f'{path}/model.ckpt', global_step=step)

    print(step, g_loss)

saver.save(sess, f'{path}/model.ckpt', global_step=step)
