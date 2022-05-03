import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle
import malaya_speech
import malaya_speech.train
from malaya_speech.train.model import univnet
from malaya_speech.train.model import universal_melgan as melgan
from malaya_speech.train.model import melgan as melgan_loss
import malaya_speech.config
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
import random

mels = glob('output-universal/mels/*.npy')
mels.extend(glob('speech-augmentation/mels/*.npy'))
random.shuffle(mels)
file_cycle = cycle(mels)


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

gen_config = univnet.GeneratorConfig()
gen_config.channel_size = 16
melgan_config = malaya_speech.config.universal_melgan_config

generator = univnet.Generator(gen_config, name='universalunivnet-generator')
discriminator = melgan.MultiScaleDiscriminator(
    melgan.WaveFormDiscriminatorConfig(
        **melgan_config['melgan_waveform_discriminator_params']
    ),
    melgan.STFTDiscriminatorConfig(
        **melgan_config['melgan_stft_discriminator_params']
    ),
    name='universalmelgan-discriminator',
)

mels_loss = melgan_loss.loss.TFMelSpectrogram()

mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()


def compute_per_example_generator_losses(audios, outputs):
    y_hat = outputs
    p_hat = discriminator(y_hat)
    p = discriminator(tf.expand_dims(audios, 2))

    adv_loss = 0.0
    for i in range(len(p_hat)):
        adv_loss += mse_loss(tf.ones_like(p_hat[i][-1]), p_hat[i][-1])
    adv_loss /= i + 1

    fm_loss = 0.0
    for i in range(len(p_hat)):
        for j in range(len(p_hat[i]) - 1):
            fm_loss += mae_loss(p[i][j], p_hat[i][j])
    fm_loss /= (i + 1) * (j + 1)
    adv_loss += 10 * fm_loss

    per_example_losses = adv_loss

    a = calculate_2d_loss(audios, tf.squeeze(y_hat, -1), loss_fn=mels_loss)

    dict_metrics_losses = {
        'adversarial_loss': adv_loss,
        'fm_loss': fm_loss,
        'gen_loss': adv_loss,
        'mels_spectrogram_loss': tf.reduce_mean(a),
    }

    return per_example_losses, dict_metrics_losses


def compute_per_example_discriminator_losses(audios, gen_outputs):
    y_hat = gen_outputs
    y = tf.expand_dims(audios, 2)
    p = discriminator(y)
    p_hat = discriminator(y_hat)

    real_loss = 0.0
    fake_loss = 0.0
    for i in range(len(p)):
        real_loss += mse_loss(tf.ones_like(p[i][-1]), p[i][-1])
        fake_loss += mse_loss(tf.zeros_like(p_hat[i][-1]), p_hat[i][-1])

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


y_hat = generator(features['mel'], training=True)
audios = features['audio']
per_example_losses, generator_losses = compute_per_example_generator_losses(
    audios, y_hat
)
generator_loss = tf.reduce_mean(per_example_losses)

y_hat = generator(features['mel'], training=True)
audios = features['audio']
per_example_losses, discriminator_losses = compute_per_example_discriminator_losses(
    audios, y_hat
)
discriminator_loss = tf.reduce_mean(per_example_losses)

for k, v in generator_losses.items():
    tf.summary.scalar(k, v)

for k, v in discriminator_losses.items():
    tf.summary.scalar(k, v)

summaries = tf.summary.merge_all()

t_vars = tf.trainable_variables()
d_vars = [
    var
    for var in t_vars
    if var.name.startswith('universalmelgan-discriminator')
]
g_vars = [
    var for var in t_vars if var.name.startswith('universalunivnet-generator')
]

d_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9).minimize(
    discriminator_loss, var_list=d_vars
)
g_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9).minimize(
    generator_loss, var_list=g_vars
)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

checkpoint = 5000
write_tensorboard = 100
epoch = 1_000_000
path = 'universal-univnet-16'

writer = tf.summary.FileWriter(f'./{path}')

ckpt_path = tf.train.latest_checkpoint(path)
if ckpt_path:
    saver.restore(sess, ckpt_path)

for i in range(epoch):
    g_loss, _ = sess.run([generator_loss, g_optimizer])
    d_loss, _ = sess.run([discriminator_loss, d_optimizer])
    s = sess.run(summaries)
    writer.add_summary(s, i)

    if i % checkpoint == 0:
        saver.save(sess, f'{path}/model.ckpt', global_step=i)

    if i % write_tensorboard == 0:
        writer.add_summary(s, i)

    print(i, g_loss, d_loss)
