import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import malaya_speech
import malaya_speech.train
import malaya_speech.config
import malaya_speech.train as train
from malaya_speech.train.model.vits import model
from malaya_speech.train.model.vits import commons
from malaya_speech.train.model.vits.model import MultiPeriodDiscriminator
from malaya_speech.train.model import vits
from malaya_speech.train.model.vits import gen
from malaya_speech.train.model import fastspeech2
from malaya_speech.train.model.fastspeech2 import model_stochastic
from librosa.filters import mel as librosa_mel_fn
from glob import glob
import random
import re

files = glob('/home/ubuntu/speech-bahasa/output-osman/mels/*.npy')
files.extend(glob('/home/ubuntu/speech-bahasa/output-osman-parliament/mels/*.npy'))
files.extend(glob('/home/ubuntu/speech-bahasa/output-osman-synthetic/mels/*.npy'))

sr = 22050
maxlen = 14 * sr
minlen = 0.2 * sr
pad_to = 8
total_steps = 500_000

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

MALAYA_SPEECH_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)

hparams = vits.HParams(**malaya_speech.config.vits_base_config)
spec_channels = hparams.data.filter_length // 2 + 1
segment_size = hparams.train.segment_size // hparams.data.hop_length

melbank = librosa_mel_fn(hparams.data.sampling_rate, hparams.data.filter_length,
                         hparams.data.n_mel_channels, hparams.data.mel_fmin, hparams.data.mel_fmax)

MEL = tf.convert_to_tensor(melbank)

f0_stat = np.load('../speech-bahasa/osman-stats/stats_f0.npy')
energy_stat = np.load('../speech-bahasa/osman-stats/stats_energy.npy')


def norm_mean_std(x, mean, std):
    zero_idxs = np.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return tf.log(tf.clip_by_value(x, clip_val, tf.reduce_max(x)) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return tf.exp(x) / C


def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output


def spectrogram_tf(audio_norm, filter_length, hop_length):
    p = int((filter_length-hop_length)/2)
    padded = tf.pad(audio_norm, [[p, p]], mode='reflect')
    spec = tf.abs(tf.signal.stft(
        padded,
        filter_length,
        hop_length,
        fft_length=None,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    ))
    spec = tf.sqrt(spec ** 2 + 1e-6)
    return spec


def spec_to_mel(spec):
    spec = tf.matmul(spec, tf.transpose(MEL))
    spec = spectral_normalize(spec)
    return spec


def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0

    return x_char.astype(np.float32)


def get_alignment(f):
    f = f"tacotron2-osman-alignment/{f.split('speech-bahasa/')[1].replace('/', '-')}"
    if os.path.exists(f):
        return np.load(f)
    else:
        return None


def generate(files):
    while True:
        random.shuffle(files)
        for f in files:
            f = f.decode() if isinstance(f, bytes) else f

            alignment = get_alignment(f)
            if alignment is None:
                continue

            wav = np.load(f.replace('mels', 'audios'))
            wav_len = len(wav)
            if wav_len > maxlen or wav_len < minlen:
                continue

            text_ids = np.load(f.replace('mels', 'text_ids'), allow_pickle=True)[
                0
            ]
            text_ids = ''.join([c for c in text_ids if c in MALAYA_SPEECH_SYMBOLS])
            text_ids = re.sub(r'[ ]+', ' ', text_ids).strip()
            text_input = np.array(
                [
                    MALAYA_SPEECH_SYMBOLS.index(c)
                    for c in text_ids
                ]
            )
            num_pad = pad_to - ((len(text_input) + 2) % pad_to)
            text_input = np.pad(
                text_input, ((1, 1)), 'constant', constant_values=((1, 2))
            )
            text_input = np.pad(
                text_input, ((0, num_pad)), 'constant', constant_values=0
            )

            f0 = np.load(f.replace('mels', 'f0s'))
            f0 = norm_mean_std(f0, f0_stat[0], f0_stat[1])
            f0 = average_by_duration(f0, alignment)
            len_f0 = [len(f0)]

            energy = np.load(f.replace('mels', 'energies'))
            energy = norm_mean_std(energy, energy_stat[0], energy_stat[1])
            energy = average_by_duration(energy, alignment)
            len_energy = [len(energy)]

            yield {
                'text_ids': text_input,
                'text_ids_len': [len(text_input)],
                'wav': wav,
                'wav_len': [wav_len],
                'f0': f0,
                'len_f0': len_f0,
                'energy': energy,
                'len_energy': len_energy,
                'alignment': alignment,
            }


def preprocess_inputs(example):
    s = spectrogram_tf(example['wav'], hparams.data.filter_length, hparams.data.hop_length)
    length = tf.cast(tf.shape(s)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['inputs'] = s
    example['inputs_length'] = length
    return example


def get_dataset(
    files,
    batch_size=20,
    thread_count=24,
):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'text_ids': tf.int32,
                'text_ids_len': tf.int32,
                'wav': tf.float32,
                'wav_len': tf.int32,
                'f0': tf.float32,
                'len_f0': tf.int32,
                'energy': tf.float32,
                'len_energy': tf.int32,
                'alignment': tf.int32,
            },
            output_shapes={
                'text_ids': tf.TensorShape([None]),
                'text_ids_len': tf.TensorShape([None]),
                'wav': tf.TensorShape([None]),
                'wav_len': tf.TensorShape([None]),
                'f0': tf.TensorShape([None]),
                'len_f0': tf.TensorShape([1]),
                'energy': tf.TensorShape([None]),
                'len_energy': tf.TensorShape([1]),
                'alignment': tf.TensorShape([None]),
            },
            args=(files,),
        )
        dataset = dataset.map(
            preprocess_inputs, num_parallel_calls=thread_count
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'inputs': tf.TensorShape([None, spec_channels]),
                'inputs_length': tf.TensorShape([None]),
                'text_ids': tf.TensorShape([None]),
                'text_ids_len': tf.TensorShape([None]),
                'wav': tf.TensorShape([None]),
                'wav_len': tf.TensorShape([None]),
                'f0': tf.TensorShape([None]),
                'len_f0': tf.TensorShape([1]),
                'energy': tf.TensorShape([None]),
                'len_energy': tf.TensorShape([1]),
                'alignment': tf.TensorShape([None]),
            },
            padding_values={
                'inputs': tf.constant(0, dtype=tf.float32),
                'inputs_length': tf.constant(0, dtype=tf.int32),
                'text_ids': tf.constant(0, dtype=tf.int32),
                'text_ids_len': tf.constant(0, dtype=tf.int32),
                'wav': tf.constant(0, dtype=tf.float32),
                'wav_len': tf.constant(0, dtype=tf.int32),
                'f0': tf.constant(0, dtype=tf.float32),
                'len_f0': tf.constant(0, dtype=tf.int32),
                'energy': tf.constant(0, dtype=tf.float32),
                'len_energy': tf.constant(0, dtype=tf.int32),
                'alignment': tf.constant(0, dtype=tf.int32),
            },
        )
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset

    return get


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = tf.reduce_mean((1-dr)**2)
        g_loss = tf.reduce_mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss, r_losses, g_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * tf.exp(-2. * logs_p)
    kl = tf.reduce_sum(kl * z_mask)
    l = kl / tf.reduce_sum(z_mask)
    return l


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = tf.stop_gradient(rl)
            loss += tf.reduce_mean(tf.abs(rl - gl))

    return loss * 2


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = tf.reduce_mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


features = get_dataset(files)().make_one_shot_iterator().get_next()
padded_features = features['inputs']
padded_lens = features['inputs_length'][:, 0]
T = features['text_ids']
T_lengths = features['text_ids_len'][:, 0]
energies = features['energy']
energies_lengths = features['len_energy'][:, 0]
f0s = features['f0']
f0s_lengths = features['len_f0'][:, 0]
alignment = features['alignment']
Y = tf.expand_dims(features['wav'], -1)
stft_shape = tf.shape(padded_features)
batch_size = tf.shape(T)[0]

model = gen.Model(len(MALAYA_SPEECH_SYMBOLS), spec_channels, segment_size, **hparams.model)
config_fs = malaya_speech.config.fastspeech2_config
config_fs = fastspeech2.Config(
    vocab_size=len(MALAYA_SPEECH_SYMBOLS), **config_fs
)
config_fs.enable_postnet = False
model_fs = model_stochastic.Model(config_fs)
mel_before, _, duration_outputs, f0_outputs, energy_outputs = model_fs(
    T, alignment, f0s, energies, training=True)

y_hat, ids_slice = model(mel_before, padded_lens)
Y_ = commons.slice_segments(Y, ids_slice * hparams.data.hop_length, hparams.train.segment_size)
outputs = model.infer(mel_before)

net_d = MultiPeriodDiscriminator(name='mp_discriminator')
y_d_hat_r, y_d_hat_g, _, _ = net_d(Y_, tf.stop_gradient(y_hat))
loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
tf.summary.scalar('loss_disc', loss_disc)

y_hat_mel = tf.TensorArray(dtype=tf.float32, size=batch_size)
init_state = (0, y_hat_mel)


def condition1(i, y_hat_mel):
    return i < batch_size


def body1(i, y_hat_mel):
    f = spectrogram_tf(y_hat[i, :, 0], hparams.data.filter_length, hparams.data.hop_length)
    f = spec_to_mel(f)
    return i + 1, y_hat_mel.write(i, f)


_, y_hat_mel = tf.while_loop(condition1, body1, init_state)
y_hat_mel = y_hat_mel.stack()
y_hat_mel.set_shape((None, None, hparams.data.n_mel_channels))

mel = tf.matmul(padded_features, tf.transpose(MEL))
mel = spectral_normalize(mel)
y_mel = commons.slice_segments(mel, ids_slice, hparams.train.segment_size // hparams.data.hop_length)

mae = tf.losses.absolute_difference
loss_mel = mae(labels=y_mel, predictions=y_hat_mel) * hparams.train.c_mel

y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(Y_, y_hat)

loss_fm = feature_loss(fmap_r, fmap_g)
loss_gen, losses_gen = generator_loss(y_d_hat_g)
loss_dur = duration_outputs

mse = tf.losses.mean_squared_error

max_length = tf.cast(tf.reduce_max(energies_lengths), tf.int32)
mask = tf.sequence_mask(
    lengths=energies_lengths, maxlen=max_length, dtype=tf.float32
)
energies_loss = mse(
    labels=energies, predictions=energy_outputs, weights=mask
)

max_length = tf.cast(tf.reduce_max(f0s_lengths), tf.int32)
mask = tf.sequence_mask(
    lengths=f0s_lengths, maxlen=max_length, dtype=tf.float32
)
f0s_loss = mse(labels=f0s, predictions=f0_outputs, weights=mask)

loss = loss_gen + loss_fm + loss_mel + loss_dur + energies_loss + f0s_loss

tf.summary.scalar('loss_gen', loss_gen)
tf.summary.scalar('loss_fm', loss_fm)
tf.summary.scalar('loss_mel', loss_mel)
tf.summary.scalar('loss_dur', loss_dur)
tf.summary.scalar('energies_loss', energies_loss)
tf.summary.scalar('f0s_loss', f0s_loss)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'mp_discriminator' in var.name]
g_vars = [var for var in t_vars if 'mp_discriminator' not in var.name]

global_step_generator = tf.Variable(
    0, trainable=False, name='global_step_generator'
)
global_step_discriminator = tf.Variable(
    0, trainable=False, name='global_step_discriminator'
)

total_steps = 1000000
init_lr = 1e-4
warmups = 60000

d_optimizer = malaya_speech.train.optimizer.adamw.create_optimizer(loss_disc,
                                                                   init_lr,
                                                                   total_steps,
                                                                   warmups,
                                                                   weight_decay_rate=0.001,
                                                                   beta_1=0.9,
                                                                   beta_2=0.98,
                                                                   epsilon=1e-6,
                                                                   tvars=d_vars,
                                                                   global_step=global_step_discriminator)

g_optimizer = malaya_speech.train.optimizer.adamw.create_optimizer(loss,
                                                                   init_lr,
                                                                   total_steps,
                                                                   warmups,
                                                                   weight_decay_rate=0.001,
                                                                   beta_1=0.9,
                                                                   beta_2=0.98,
                                                                   epsilon=1e-6,
                                                                   tvars=g_vars,
                                                                   global_step=global_step_generator)

summaries = tf.summary.merge_all()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

checkpoint = 1000
write_tensorboard = 100
path = 'e2e-fastspeech2-osman-v3'

writer = tf.summary.FileWriter(f'./{path}')

ckpt_path = tf.train.latest_checkpoint(path)
if ckpt_path:
    saver.restore(sess, ckpt_path)

step = sess.run(global_step_discriminator)
while step < total_steps:

    step, d_loss, _ = sess.run([global_step_discriminator, loss_disc, d_optimizer])
    g_step, input_shape, g_loss, _ = sess.run([global_step_generator, stft_shape, (loss_gen, loss_fm,
                                                                                   loss_mel, loss_dur, energies_loss, f0s_loss), g_optimizer])
    s = sess.run(summaries)

    if step % checkpoint == 0:
        saver.save(sess, f'{path}/model.ckpt', global_step=step)

    if step % write_tensorboard == 0:
        writer.add_summary(s, step)

    print(step, input_shape, g_loss, d_loss)
