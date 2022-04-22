import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle
import re
import json

with open('mels-male-v2.json') as fopen:
    files = json.load(fopen)

reduction_factor = 1
maxlen = 904
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


def generate(files):
    file_cycle = cycle(files)
    while True:
        f = next(file_cycle).decode()
        mel = np.load(f)
        mel_length = len(mel)
        f_wav = f.replace('/mels', '/audios')
        wav = np.load(f_wav)
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
        num_pad = pad_to - len(wav) % pad_to
        wav = np.pad(
            wav,
            ((0, num_pad)),
            'constant',
        )
        len_mel = [len(mel)]
        len_text_ids = [len(text_input)]

        yield {
            'mel': mel,
            'text_ids': text_input,
            'len_mel': len_mel,
            'len_text_ids': len_text_ids,
            'f': [f],
            'audio': wav,
        }


def get_dataset(files, batch_size=16, shuffle_size=32, thread_count=24):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'text_ids': tf.int32,
                'len_mel': tf.int32,
                'len_text_ids': tf.int32,
                'f': tf.string,
                'audio': tf.float32,
            },
            output_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'f': tf.TensorShape([1]),
                'audio': tf.TensorShape([None]),
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
                'audio': tf.TensorShape([None]),
            },
            padding_values={
                'mel': tf.constant(0, dtype=tf.float32),
                'text_ids': tf.constant(0, dtype=tf.int32),
                'len_mel': tf.constant(0, dtype=tf.int32),
                'len_text_ids': tf.constant(0, dtype=tf.int32),
                'f': tf.constant('', dtype=tf.string),
                'audio': tf.constant(0, dtype=tf.float32),
            },
        )
        return dataset

    return get


features = get_dataset(files['train'])().make_one_shot_iterator().get_next()
features

import malaya_speech
import malaya_speech.train
from malaya_speech.train.model import vits, melgan, hifigan
from malaya_speech.train.model.vits.slicing import rand_slice_segments
from malaya_speech.train.model import stft
import malaya_speech.config
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
import malaya_speech.train as train

segment_size = 8192
hop_size = 256
config = vits.Config(mel=80, vocabs=len(MALAYA_SPEECH_SYMBOLS))
model = vits.Model(config)

melgan_config = malaya_speech.config.melgan_config
generator = melgan.Generator(
    melgan.GeneratorConfig(**melgan_config['melgan_generator_params']),
    name='melgan-generator',
)
discriminator = melgan.MultiScaleDiscriminator(
    melgan.DiscriminatorConfig(**melgan_config['melgan_discriminator_params']),
    name='melgan-discriminator',
)

mels_loss = melgan.loss.TFMelSpectrogram()
mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()


text = features['text_ids']
text_lengths = features['len_text_ids'][:, 0]
mel_outputs = features['mel']
mel_lengths = features['len_mel'][:, 0]
wavs = features['audio']

losses, attn, latent, z_slice, ids_slice = model.compute_loss(text=text,
                                                              textlen=text_lengths,
                                                              mel=mel_outputs,
                                                              mellen=mel_lengths)

mel = vits.slice_segments(mel_outputs, ids_slice, model.segment_size // model.hop_size, np.log(1e-2))
y = vits.slice_segments(tf.expand_dims(wavs, -1), ids_slice * model.hop_size, model.segment_size)


def compute_per_example_discriminator_losses():

    y_hat = generator(z_slice, training=True)
    p = discriminator(y)
    p_hat = discriminator(y_hat)

    real_loss = 0.0
    fake_loss = 0.0
    for i in range(len(p)):
        real_loss += calculate_3d_loss(
            tf.ones_like(p[i][-1]), p[i][-1], loss_fn=mse_loss
        )
        fake_loss += calculate_3d_loss(
            tf.zeros_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=mse_loss
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


def compute_per_example_generator_losses():

    y_hat = generator(z_slice, training=True)
    p = discriminator(y)
    p_hat = discriminator(y_hat)

    adv_loss = 0.0
    for i in range(len(p_hat)):
        adv_loss += calculate_3d_loss(
            tf.ones_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=mse_loss
        )
    adv_loss /= i + 1

    fm_loss = 0.0
    for i in range(len(p_hat)):
        for j in range(len(p_hat[i]) - 1):
            fm_loss += calculate_3d_loss(
                p[i][j], p_hat[i][j], loss_fn=mae_loss
            )
    fm_loss /= (i + 1) * (j + 1)

    a = calculate_2d_loss(tf.squeeze(y, -1), tf.squeeze(y_hat, -1), loss_fn=mels_loss)
    mels_spectrogram_loss = tf.reduce_mean(a)
    mel_loss = calculate_3d_loss(mel, z_slice, loss_fn=mae_loss) * 45.0
    kl_loss = losses['kl'] * 1.0
    per_example_losses = adv_loss + fm_loss * 10 + mel_loss + kl_loss + losses['durloss']
    dict_metrics_losses = {
        'adversarial_loss': adv_loss,
        'fm_loss': fm_loss,
        'mels_spectrogram_loss': mels_spectrogram_loss,
        'mel_loss': mel_loss,
        'kl': losses['kl'],
        'durloss': losses['durloss'],
    }

    return per_example_losses, dict_metrics_losses


per_example_losses, generator_losses = compute_per_example_generator_losses()
generator_loss = tf.reduce_mean(per_example_losses)
per_example_losses, discriminator_losses = compute_per_example_discriminator_losses()
discriminator_loss = tf.reduce_mean(per_example_losses)

for k, v in generator_losses.items():
    tf.summary.scalar(k, v)

for k, v in discriminator_losses.items():
    tf.summary.scalar(k, v)

tf.summary.scalar('discriminator_loss', discriminator_loss)
tf.summary.scalar('generator_loss', generator_loss)


t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if var.name.startswith('melgan-discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('melgan-generator')]
glowtts_vars = [var for var in t_vars if not var.name.startswith(
    'melgan-generator') and not var.name.startswith('melgan-discriminator')]
g_vars = g_vars + glowtts_vars

checkpoint = 2500
epoch = 200000
path = 'vits-male-melgan'

global_step_generator = tf.Variable(
    0, trainable=False, name='global_step_generator'
)
global_step_discriminator = tf.Variable(
    0, trainable=False, name='global_step_discriminator'
)

g_optimizer = train.optimizer.adamw.create_optimizer(
    generator_loss,
    init_lr=1e-4,
    num_train_steps=epoch,
    num_warmup_steps=10000,
    end_learning_rate=1e-6,
    weight_decay_rate=0.01,
    beta_1=0.8,
    beta_2=0.99,
    epsilon=1e-9,
    clip_norm=1.0,
    tvars=g_vars,
    global_step=global_step_generator,
)

d_optimizer = train.optimizer.adamw.create_optimizer(
    discriminator_loss,
    init_lr=1e-4,
    num_train_steps=epoch,
    num_warmup_steps=10000,
    end_learning_rate=1e-6,
    weight_decay_rate=0.01,
    beta_1=0.8,
    beta_2=0.99,
    epsilon=1e-9,
    clip_norm=1.0,
    tvars=d_vars,
    global_step=global_step_discriminator,
)

summaries = tf.summary.merge_all()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
writer = tf.summary.FileWriter(f'./{path}')

ckpt_path = tf.train.latest_checkpoint(path)
if ckpt_path:
    saver.restore(sess, ckpt_path)
    print(f'restoring checkpoint from {ckpt_path}')

for i in range(0, epoch):
    g_losses, g_loss, _ = sess.run([generator_losses, generator_loss, g_optimizer])
    d_losses, d_loss, _ = sess.run([discriminator_losses, discriminator_loss, d_optimizer])

    if i % 100 == 0:
        s = sess.run(summaries)
        writer.add_summary(s, i)

    if i % checkpoint == 0:
        saver.save(sess, f'{path}/model.ckpt', global_step=i)

    print('generator', i, g_loss, g_losses)
    print('discriminator', i, d_loss, d_losses)
    print()

saver.save(sess, f'{path}/model.ckpt', global_step=epoch)
