import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import random
import tensorflow as tf
from math import ceil
from glob import glob
from pysptk import sptk
from collections import defaultdict
from malaya_speech.train.model import speechsplit
from malaya_speech import train
from scipy.signal import get_window
from scipy import signal
import pandas as pd
import malaya_speech
import sklearn


def butter_highpass(cutoff, fs, order = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = 'high', analog = False)
    return b, a


vggvox_v2 = malaya_speech.gender.deep_model(model = 'vggvox-v2')
speaker_model = malaya_speech.speaker_vector.deep_model('vggvox-v2')

vctk = glob('vtck/**/*.flac', recursive = True)

vctk_speakers = defaultdict(list)
for f in vctk:
    s = f.split('/')[-1].split('_')[0]
    vctk_speakers[s].append(f)

files = glob('/home/husein/speech-bahasa/ST-CMDS-20170001_1-OS/*.wav')
speakers_mandarin = defaultdict(list)
for f in files:
    speakers_mandarin[f[:-9]].append(f)

df_nepali = pd.read_csv(
    '/home/husein/speech-bahasa/nepali_0/asr_nepali/utt_spk_text.tsv',
    sep = '\t',
    header = None,
)
asr_nepali = glob('/home/husein/speech-bahasa/*/asr_nepali/data/*/*.flac')
asr_nepali_replaced = {
    f.split('/')[-1].replace('.flac', ''): f for f in asr_nepali
}
df_nepali = df_nepali[df_nepali[0].isin(asr_nepali_replaced.keys())]

speakers_nepali = defaultdict(list)
for i in range(len(df_nepali)):
    speakers_nepali[df_nepali.iloc[i, 1]].append(
        asr_nepali_replaced[df_nepali.iloc[i, 0]]
    )

speakers = []
for s in vctk_speakers.keys():
    speakers.extend(
        random.sample(vctk_speakers[s], min(500, len(vctk_speakers[s])))
    )

for s in speakers_mandarin.keys():
    speakers.extend(
        random.sample(speakers_mandarin[s], min(200, len(speakers_mandarin[s])))
    )

for s in speakers_nepali.keys():
    speakers.extend(
        random.sample(speakers_nepali[s], min(200, len(speakers_nepali[s])))
    )

salina = glob('/home/husein/speech-bahasa/salina/output-wav-salina/*.wav')
salina = random.sample(salina, 5000)
male = glob('/home/husein/speech-bahasa/turki/output-wav-turki/*.wav')
male.extend(
    glob(
        '/home/husein/speech-bahasa/dari-pasentran-ke-istana/output-wav-dari-pasentran-ke-istana/*.wav'
    )
)
male = random.sample(male, 5000)
haqkiem = glob('/home/husein/speech-bahasa/haqkiem/*.wav')
khalil = glob('/home/husein/speech-bahasa/tolong-sebut/*.wav')
mas = glob('/home/husein/speech-bahasa/sebut-perkataan-woman/*.wav')
husein = glob('/home/husein/speech-bahasa/audio-wattpad/*.wav')
husein.extend(glob('/home/husein/speech-bahasa/audio-iium/*.wav'))
husein.extend(glob('/home/husein/speech-bahasa/audio/*.wav'))
husein.extend(glob('/home/husein/speech-bahasa/sebut-perkataan-man/*.wav'))

files = salina + male + haqkiem + khalil + mas + husein + speakers
sr = 22050
freqs = {'female': [100, 600], 'male': [50, 250]}
b, a = butter_highpass(30, sr, order = 5)


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    f0 = f0.astype(float).copy()
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -3, 4)
    return f0


def preprocess_wav(x):
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis = 0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (np.random.uniform(size = y.shape[0]) - 0.5) * 1e-06
    return wav


def get_f0(wav, lo, hi):
    f0_rapt = sptk.rapt(
        wav.astype(np.float32) * 32768, sr, 256, min = lo, max = hi, otype = 2
    )
    index_nonzero = f0_rapt != -1e10
    mean_f0, std_f0 = (
        np.mean(f0_rapt[index_nonzero]),
        np.std(f0_rapt[index_nonzero]),
    )
    return speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)


def pad_seq(x, base = 8):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), x.shape[0]


def generate(hop_size = 256):
    while True:
        shuffled = sklearn.utils.shuffle(files)
        for f in shuffled:
            x, fs = malaya_speech.load(f, sr = sr)
            wav = preprocess_wav(x)
            x_16k = malaya_speech.resample(x, sr, 16000)
            lo, hi = freqs.get(vggvox_v2(x_16k), [50, 250])
            f0 = np.expand_dims(get_f0(wav, lo, hi), -1)
            mel = malaya_speech.featurization.universal_mel(wav)

            batch_max_steps = random.randint(16384, 55125)
            batch_max_frames = batch_max_steps // hop_size

            if len(mel) > batch_max_frames:
                interval_start = 0
                interval_end = len(mel) - batch_max_frames
                start_frame = random.randint(interval_start, interval_end)
                start_step = start_frame * hop_size
                mel = mel[start_frame : start_frame + batch_max_frames, :]
                f0 = f0[start_frame : start_frame + batch_max_frames, :]
                wav = wav[start_step : start_step + batch_max_steps]

            mel, _ = pad_seq(mel)
            f0, _ = pad_seq(f0)

            wav_16k = malaya_speech.resample(wav, sr, 16000)
            v = speaker_model([wav_16k])[0]
            v = v / v.max()

            yield {
                'mel': mel,
                'mel_length': [len(mel)],
                'f0': f0,
                'f0_length': [len(f0)],
                'audio': wav,
                'v': v,
            }


def get_dataset(batch_size = 4):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'mel_length': tf.int32,
                'f0': tf.float32,
                'f0_length': tf.int32,
                'audio': tf.float32,
                'v': tf.float32,
            },
            output_shapes = {
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'f0': tf.TensorShape([None, 1]),
                'f0_length': tf.TensorShape([None]),
                'audio': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
        )
        dataset = dataset.shuffle(batch_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'audio': tf.TensorShape([None]),
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'f0': tf.TensorShape([None, 1]),
                'f0_length': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
            padding_values = {
                'audio': tf.constant(0, dtype = tf.float32),
                'mel': tf.constant(0, dtype = tf.float32),
                'mel_length': tf.constant(0, dtype = tf.int32),
                'f0': tf.constant(0, dtype = tf.float32),
                'f0_length': tf.constant(0, dtype = tf.int32),
                'v': tf.constant(0, dtype = tf.float32),
            },
        )
        return dataset

    return get


total_steps = 500000


def model_fn(features, labels, mode, params):
    vectors = features['v']
    X = features['mel']
    len_X = features['mel_length'][:, 0]
    X_f0 = features['f0']
    len_X_f0 = features['f0_length'][:, 0]
    hparams = speechsplit.hparams
    interplnr = speechsplit.InterpLnr(hparams)
    model = speechsplit.Model(hparams)
    model_F0 = speechsplit.Model_F0(hparams)

    bottleneck_speaker = tf.keras.layers.Dense(hparams.dim_spk_emb)
    speaker_dim = bottleneck_speaker(vectors)

    x_f0_intrp = interplnr(tf.concat([X, X_f0], axis = -1), len_X)
    f0_org_intrp = speechsplit.quantize_f0_tf(x_f0_intrp[:, :, -1])
    x_f0_intrp_org = tf.concat((x_f0_intrp[:, :, :-1], f0_org_intrp), axis = -1)
    f0_org = speechsplit.quantize_f0_tf(X_f0[:, :, 0])

    _, _, _, _, mel_outputs = model(x_f0_intrp_org, X, speaker_dim)
    _, _, _, f0_outputs = model_F0(X, f0_org)

    loss_f = tf.losses.absolute_difference
    max_length = tf.cast(tf.reduce_max(len_X), tf.int32)
    mask = tf.sequence_mask(
        lengths = len_X, maxlen = max_length, dtype = tf.float32
    )
    mask = tf.expand_dims(mask, axis = -1)
    mel_loss = loss_f(labels = X, predictions = mel_outputs, weights = mask)
    f0_loss = loss_f(labels = f0_org, predictions = f0_outputs, weights = mask)

    loss = mel_loss + f0_loss

    tf.identity(loss, 'total_loss')
    tf.identity(mel_loss, 'mel_loss')
    tf.identity(f0_loss, 'f0_loss')

    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('mel_loss', mel_loss)
    tf.summary.scalar('f0_loss', f0_loss)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value = 1e-4, shape = [], dtype = tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        total_steps,
        end_learning_rate = 1e-6,
        power = 1.0,
        cycle = False,
    )
    tf.summary.scalar('learning_rate', learning_rate)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['total_loss', 'mel_loss', 'f0_loss'], every_n_iter = 1
    )
]
train_dataset = get_dataset()

save_directory = 'speechsplit-vggvox-v2'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 2000,
    max_steps = total_steps,
    train_hooks = train_hooks,
    eval_step = 0,
)
