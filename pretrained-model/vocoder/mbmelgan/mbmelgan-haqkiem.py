import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle

mels = glob('../speech-bahasa/output-haqkiem/mels/*.npy')
file_cycle = cycle(mels)
f = next(file_cycle)

import random


def generate(batch_max_steps = 8192, hop_size = 256):
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
            audio = audio[start_step : start_step + batch_max_steps]
            mel = mel[start_frame : start_frame + batch_max_frames, :]
        else:
            audio = np.pad(audio, [[0, batch_max_steps - len(audio)]])
            mel = np.pad(mel, [[0, batch_max_frames - len(mel)], [0, 0]])

        yield {'mel': mel, 'audio': audio}


dataset = tf.data.Dataset.from_generator(
    generate,
    {'mel': tf.float32, 'audio': tf.float32},
    output_shapes = {
        'mel': tf.TensorShape([None, 80]),
        'audio': tf.TensorShape([None]),
    },
)
dataset = dataset.shuffle(32)
dataset = dataset.padded_batch(
    32,
    padded_shapes = {
        'audio': tf.TensorShape([None]),
        'mel': tf.TensorShape([None, 80]),
    },
    padding_values = {
        'audio': tf.constant(0, dtype = tf.float32),
        'mel': tf.constant(0, dtype = tf.float32),
    },
)

features = dataset.make_one_shot_iterator().get_next()

import malaya_speech
import malaya_speech.train
import malaya_speech.config
from malaya_speech.train.model import melgan, mb_melgan, stft
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss

mb_melgan_config = malaya_speech.config.mb_melgan_config
generator = melgan.Generator(
    mb_melgan.GeneratorConfig(**mb_melgan_config['melgan_generator_params']),
    name = 'mb_melgan-generator',
)
pqmf = mb_melgan.PQMF(
    mb_melgan.GeneratorConfig(**mb_melgan_config['melgan_generator_params']),
    dtype = tf.float32,
    name = 'pqmf',
)
discriminator = melgan.MultiScaleDiscriminator(
    mb_melgan.DiscriminatorConfig(
        **mb_melgan_config['melgan_discriminator_params']
    ),
    name = 'melgan-discriminator',
)

mels_loss = melgan.loss.TFMelSpectrogram()


mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()


sub_band_stft_loss = stft.loss.MultiResolutionSTFT(
    **mb_melgan_config['subband_stft_loss_params']
)

full_band_stft_loss = stft.loss.MultiResolutionSTFT(
    **mb_melgan_config['stft_loss_params']
)


def compute_per_example_generator_losses(features):
    y_mb_hat = generator(features['mel'], training = True)
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

    sub_sc_loss = tf.reduce_mean(
        tf.reshape(sub_sc_loss, [-1, pqmf.subbands]), -1
    )
    sub_mag_loss = tf.reduce_mean(
        tf.reshape(sub_mag_loss, [-1, pqmf.subbands]), -1
    )
    full_sc_loss, full_mag_loss = calculate_2d_loss(
        audios, tf.squeeze(y_hat, -1), full_band_stft_loss
    )

    generator_loss = 0.5 * (sub_sc_loss + sub_mag_loss) + 0.5 * (
        full_sc_loss + full_mag_loss
    )

    p_hat = discriminator(y_hat)
    p = discriminator(tf.expand_dims(audios, 2))

    adv_loss = 0.0
    for i in range(len(p_hat)):
        adv_loss += calculate_3d_loss(
            tf.ones_like(p_hat[i][-1]), p_hat[i][-1], loss_fn = mse_loss
        )
    adv_loss /= i + 1

    generator_loss += 2.5 * adv_loss

    per_example_losses = generator_loss

    a = calculate_2d_loss(audios, tf.squeeze(y_hat, -1), loss_fn = mels_loss)

    dict_metrics_losses = {
        'adversarial_loss': adv_loss,
        'gen_loss': tf.reduce_mean(generator_loss),
        'subband_spectral_convergence_loss': tf.reduce_mean(sub_sc_loss),
        'subband_log_magnitude_loss': tf.reduce_mean(sub_mag_loss),
        'fullband_spectral_convergence_loss': tf.reduce_mean(full_sc_loss),
        'fullband_log_magnitude_loss': tf.reduce_mean(full_mag_loss),
        'mels_spectrogram_loss': tf.reduce_mean(a),
    }

    return per_example_losses, dict_metrics_losses


def compute_per_example_discriminator_losses(features):
    y_mb_hat = generator(features['mel'], training = True)
    audios = features['audio']
    y_hat = pqmf.synthesis(y_mb_hat)
    y = tf.expand_dims(audios, 2)
    p = discriminator(y)
    p_hat = discriminator(y_hat)

    real_loss = 0.0
    fake_loss = 0.0
    for i in range(len(p)):
        real_loss += calculate_3d_loss(
            tf.ones_like(p[i][-1]), p[i][-1], loss_fn = mse_loss
        )
        fake_loss += calculate_3d_loss(
            tf.zeros_like(p_hat[i][-1]), p_hat[i][-1], loss_fn = mse_loss
        )
    real_loss /= i + 1
    fake_loss /= i + 1
    dis_loss = real_loss + fake_loss

    per_example_losses = dis_loss

    dict_metrics_losses = {
        'real_loss': real_loss,
        'fake_loss': fake_loss,
        'dis_loss': dis_loss,
    }

    return per_example_losses, dict_metrics_losses


per_example_losses, generator_losses = compute_per_example_generator_losses(
    features
)
generator_loss = tf.reduce_mean(per_example_losses)

per_example_losses, discriminator_losses = compute_per_example_discriminator_losses(
    features
)
discriminator_loss = tf.reduce_mean(per_example_losses)

for k, v in generator_losses.items():
    tf.summary.scalar(k, v)

for k, v in discriminator_losses.items():
    tf.summary.scalar(k, v)

summaries = tf.summary.merge_all()

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if var.name.startswith('melgan-discriminator')]
g_vars = [
    var
    for var in t_vars
    if var.name.startswith('mb_melgan-generator') or var.name.startswith('pqmf')
]

d_optimizer = tf.train.AdamOptimizer(0.0001, beta1 = 0.5, beta2 = 0.9).minimize(
    discriminator_loss, var_list = d_vars
)
g_optimizer = tf.train.AdamOptimizer(0.0001, beta1 = 0.5, beta2 = 0.9).minimize(
    generator_loss, var_list = g_vars
)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list = g_vars)
saver.restore(sess, tf.train.latest_checkpoint('mbmelgan-haqkiem'))

saver = tf.train.Saver()

checkpoint = 5000
epoch = 1_100_000
path = 'mbmelgan-haqkiem-combined'

writer = tf.summary.FileWriter(f'./{path}')

ckpt_path = tf.train.latest_checkpoint(path)
if ckpt_path:
    saver.restore(sess, ckpt_path)
    print(f'restoring checkpoint from {ckpt_path}')

for i in range(0, epoch):
    g_loss, _ = sess.run([generator_loss, g_optimizer])
    d_loss, _ = sess.run([discriminator_loss, d_optimizer])
    s = sess.run(summaries)
    writer.add_summary(s, i)

    if i % checkpoint == 0:
        saver.save(sess, f'{path}/model.ckpt', global_step = i)

    print(i, g_loss, d_loss)

saver.save(sess, f'{path}/model.ckpt', global_step = epoch)
