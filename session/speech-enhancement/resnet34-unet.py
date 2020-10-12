import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import librosa
from glob import glob
import soundfile as sf
import random
from pysndfx import AudioEffectsChain
from scipy.special import expit
from itertools import cycle

files = glob('../speech-bahasa/LibriSpeech/*/*/*/*.flac') + glob(
    '../youtube/clean-wav/*.wav'
)

# files = glob('../youtube/clean-wav/*.wav')
files = list(set(files))
random.shuffle(files)
print(len(files))

file_cycle = cycle(files)

import pickle

with open('../youtube/ambients.pkl', 'rb') as fopen:
    ambient = pickle.load(fopen)


def sox_reverb(
    y, reverberance = 1, hf_damping = 1, room_scale = 1, stereo_depth = 1
):
    apply_audio_effects = AudioEffectsChain().reverb(
        reverberance = reverberance,
        hf_damping = hf_damping,
        room_scale = room_scale,
        stereo_depth = stereo_depth,
        pre_delay = 20,
        wet_gain = 0,
        wet_only = False,
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def sox_augment_low(
    y,
    min_bass_gain = 5,
    reverberance = 1,
    hf_damping = 1,
    room_scale = 1,
    stereo_depth = 1,
    negate = 1,
):
    if negate:
        min_bass_gain = -min_bass_gain
    apply_audio_effects = (
        AudioEffectsChain()
        .lowshelf(gain = min_bass_gain, frequency = 300, slope = 0.1)
        .reverb(
            reverberance = reverberance,
            hf_damping = hf_damping,
            room_scale = room_scale,
            stereo_depth = stereo_depth,
            pre_delay = 20,
            wet_gain = 0,
            wet_only = False,
        )
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def sox_augment_high(
    y,
    min_bass_gain = 5,
    reverberance = 1,
    hf_damping = 1,
    room_scale = 1,
    stereo_depth = 1,
    negate = 1,
):
    if negate:
        min_bass_gain = -min_bass_gain
    apply_audio_effects = (
        AudioEffectsChain()
        .highshelf(
            gain = -min_bass_gain * (1 - expit(np.max(y))),
            frequency = 300,
            slope = 0.1,
        )
        .reverb(
            reverberance = reverberance,
            hf_damping = hf_damping,
            room_scale = room_scale,
            stereo_depth = stereo_depth,
            pre_delay = 20,
            wet_gain = 0,
            wet_only = False,
        )
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def sox_augment_combine(
    y,
    min_bass_gain_low = 5,
    min_bass_gain_high = 5,
    reverberance = 1,
    hf_damping = 1,
    room_scale = 1,
    stereo_depth = 1,
):

    apply_audio_effects = (
        AudioEffectsChain()
        .lowshelf(gain = min_bass_gain_low, frequency = 300, slope = 0.1)
        .highshelf(gain = -min_bass_gain_high, frequency = 300, slope = 0.1)
        .reverb(
            reverberance = reverberance,
            hf_damping = hf_damping,
            room_scale = room_scale,
            stereo_depth = stereo_depth,
            pre_delay = 20,
            wet_gain = 0,
            wet_only = False,
        )
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def fftnoise(f):
    f = np.array(f, dtype = 'complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def add_band_limited_noise(
    sample, min_freq = 200, max_freq = 24000, samplerate = 16000
):
    freqs = np.abs(np.fft.fftfreq(len(sample), 1 / samplerate))
    f = np.zeros(len(sample))
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return sample + fftnoise(f)


def add_uniform_noise(sample, power = 0.01):
    y_noise = sample.copy()
    noise_amp = power * np.random.uniform() * np.amax(y_noise)
    return y_noise.astype('float64') + noise_amp * np.random.normal(
        size = y_noise.shape[0]
    )


def random_sampling(sample, sr, length = 500):
    sr = int(sr / 1000)
    up = len(sample) - (sr * length)
    if up < 1:
        r = 0
    else:
        r = np.random.randint(0, up)
    return sample[r : r + sr * length]


def add_noise(sample, noise, random_sample = True, factor = 0.1):
    y_noise = sample.copy()
    if len(y_noise) > len(noise):
        noise = np.tile(noise, int(np.ceil(len(y_noise) / len(noise))))
    else:
        if random_sample:
            noise = noise[np.random.randint(0, len(noise) - len(y_noise) + 1) :]
    return y_noise + noise[: len(y_noise)] * factor


def read_flac(file):
    data, old_samplerate = sf.read(file)
    if len(data.shape) == 2:
        data = data[:, 0]
    return data, old_samplerate


def read_wav(file):
    y, sr = librosa.load(file, sr = None)
    return y, sr


def read_file(file):
    if '.flac' in file:
        y, sr = read_flac(file)
    if '.wav' in file:
        y, sr = read_wav(file)
    return y, sr


def combine_speakers(files, n = 5):
    w_samples = random.sample(files, n)
    y = [w_samples[0]]
    left = w_samples[0]
    for i in range(1, n):

        right = w_samples[i]

        overlap = random.uniform(0.01, 1.5)
        left_len = int(overlap * len(left))

        padded_right = np.pad(right, (left_len, 0))

        if len(left) > len(padded_right):
            padded_right = np.pad(
                padded_right, (0, len(left) - len(padded_right))
            )
        else:
            left = np.pad(left, (0, len(padded_right) - len(left)))

        y.append(padded_right)
        left = left + padded_right
    return left, y


def scale_mel(
    y,
    sr = 16000,
    n_fft = 2048,
    hop_length = 100,
    win_length = 1000,
    n_mels = 256,
    ref_db = 20,
    max_db = 100,
    factor = 15,
    scale = True,
):
    mel = librosa.feature.melspectrogram(
        y = y,
        sr = sr,
        S = None,
        n_fft = n_fft,
        hop_length = hop_length,
        win_length = win_length,
        window = 'hann',
        center = True,
        pad_mode = 'reflect',
        power = 1.0,
        n_mels = n_mels,
    )
    if scale:
        mel = factor * np.log10(mel)
        mel = np.clip((mel - ref_db + max_db) / max_db, 1e-11, 1)
    return mel


def scale_spectrogram(
    y,
    sr = 16000,
    n_fft = 2048,
    hop_length = 100,
    win_length = 1000,
    n_mels = 256,
    ref_db = 20,
    max_db = 100,
    factor = 9.5,
    scale = True,
):
    D = librosa.stft(
        y,
        n_fft = n_fft,
        hop_length = hop_length,
        win_length = win_length,
        window = 'hann',
        center = True,
        dtype = None,
        pad_mode = 'reflect',
    )
    spectrogram, _ = librosa.magphase(D, power = 1)

    if scale:
        spectrogram = factor * np.log10(spectrogram)
        spectrogram = np.clip(
            (spectrogram - ref_db + max_db) / max_db, 1e-11, 1
        )
    return spectrogram


def unscale_mel(mel, ref_db = 20, max_db = 100, factor = 15):
    inv_mel = ((mel * max_db) - max_db + ref_db) / factor
    inv_mel = np.power(10, inv_mel)
    inv_mel[inv_mel < 5e-6] = 0.0
    return inv_mel


def unscale_spectrogram(spectrogram, ref_db = 20, max_db = 100, factor = 9.5):
    inv_spectrogram = ((spectrogram * max_db) - max_db + ref_db) / factor
    inv_spectrogram = np.power(10, inv_spectrogram)
    return inv_spectrogram


def mel_to_spectrogram(mel, sr = 16000, n_fft = 2048):
    return librosa.feature.inverse.mel_to_stft(
        mel, sr = sr, n_fft = n_fft, power = 1.0
    )


def to_signal(spectrogram, n_iter = 10, win_length = 1000, hop_length = 100):
    return librosa.griffinlim(
        spectrogram,
        n_iter = n_iter,
        win_length = win_length,
        hop_length = hop_length,
    )


def sampling(combined, frame_duration_ms = 700, sample_rate = 16000):
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    results = []
    while offset + n <= len(combined):
        results.append(combined[offset : offset + n])
        offset += n
    if offset < len(combined):
        results.append(combined[offset:])

    return results


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    MaxPooling2D,
    Dropout,
    concatenate,
    UpSampling2D,
)
import tensorflow as tf
import segmentation_models as sm
import mp


def calc(signal):

    choice = random.randint(0, 5)
    if choice == 0:
        x = sox_augment_high(
            signal,
            min_bass_gain = random.randint(25, 50),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = 1,
        )
    if choice == 1:
        x = sox_augment_high(
            signal,
            min_bass_gain = random.randint(25, 70),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = 0,
        )
    if choice == 2:
        x = sox_augment_low(
            signal,
            min_bass_gain = random.randint(5, 30),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = random.randint(0, 1),
        )
    if choice == 3:
        x = sox_augment_combine(
            signal,
            min_bass_gain_high = random.randint(25, 70),
            min_bass_gain_low = random.randint(5, 30),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 90),
        )
    if choice == 4:
        x = sox_reverb(
            signal,
            reverberance = random.randint(10, 80),
            hf_damping = 10,
            room_scale = random.randint(10, 90),
        )
    if choice == 5:
        x = signal

    if random.randint(0, 1):
        x = add_uniform_noise(x, power = random.uniform(0.005, 0.01))

    return x


def calc_mel(signal):
    x, signal = signal
    y = scale_mel(signal)
    x = scale_mel(x)
    z = y.T - x.T
    return x.T, z


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(calc(f))
    return results


def loop_mel(files):
    files = files[0]
    results = []
    for f in files:
        results.append(calc_mel(f))
    return results


def generate(batch_size = 100, core = 16, repeat = 2):
    while True:
        batch_files = [next(file_cycle) for _ in range(batch_size)]
        print(batch_files)
        print('before wavs')
        wavs = [read_file(f)[0] for f in batch_files]
        print('after wavs')

        samples = []
        print('before iterating wavs')
        for wav in wavs:
            if random.random() < 0.7:
                signal = wav.copy()

                if random.randint(0, 1):
                    signal_ = random.choice(wavs)
                    signal = add_noise(
                        signal, signal_, factor = random.uniform(0.6, 1.0)
                    )

            else:
                r = random.randint(2, 6)
                signal = combine_speakers(wavs, min(len(wavs), r))[0]

            signal = random_sampling(
                signal, 16000, random.randint(60000, 200000)
            )

            samples.append(signal)

        R = []
        for s in samples:
            if random.random() > 0.8:
                signal_ = random.choice(ambient)
                s = add_noise(s, signal_, factor = random.uniform(0.1, 0.3))
            R.append(s)

        print('len samples', len(samples))
        results = mp.multiprocessing(R, loop, cores = min(len(samples), core))
        print('after len samples', len(samples))

        X, Y = [], []
        for o in range(len(samples)):
            X.extend(sampling(results[o], 1200))
            Y.extend(sampling(samples[o], 1200))
        # return samples, results, X, Y

        combined = list(zip(X, Y))
        print('len combined', len(combined))
        results = mp.multiprocessing(
            combined, loop_mel, cores = min(len(combined), core)
        )
        print('after len combined', len(combined))

        for _ in range(repeat):
            #     random.shuffle(results)
            for r in results:
                yield {'inputs': r[0], 'targets': r[1]}


def get_dataset(batch_size = 32, shuffle_size = 128, prefetch_size = 128):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'inputs': tf.float32, 'targets': tf.float32},
            output_shapes = {
                'inputs': tf.TensorShape([None, 256]),
                'targets': tf.TensorShape([None, 256]),
            },
        )
        dataset = dataset.shuffle(shuffle_size)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'inputs': tf.TensorShape([256, 256]),
                'targets': tf.TensorShape([256, 256]),
            },
            padding_values = {
                'inputs': tf.constant(0, dtype = tf.float32),
                'targets': tf.constant(0, dtype = tf.float32),
            },
        )
        dataset = dataset.prefetch(prefetch_size)
        return dataset

    return get


init_lr = 1e-4
epochs = 500000


def model_fn(features, labels, mode, params):
    model = sm.Unet(
        'resnet34',
        classes = 1,
        activation = 'tanh',
        input_shape = (256, 256, 1),
        encoder_weights = None,
    )
    Y = tf.expand_dims(features['inputs'], -1)
    logits = model(Y)
    loss = tf.compat.v1.losses.huber_loss(
        features['targets'], logits[:, :, :, 0]
    )
    loss = tf.reduce_mean(loss)
    tf.identity(loss, 'train_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(
            value = init_lr, shape = [], dtype = tf.float32
        )
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            epochs,
            end_learning_rate = 1e-6,
            power = 1.0,
            cycle = False,
        )
        #         optimizer = tf.train.RMSPropOptimizer(
        #             learning_rate, decay = 0.9, momentum = 0.9, epsilon = 1.0
        #         )
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


import malaya_speech.train as train

train_hooks = [tf.train.LoggingTensorHook(['train_loss'], every_n_iter = 1)]
train_dataset = get_dataset(batch_size = 16)

save_directory = 'output-resnet34-unet'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 3,
    log_step = 1,
    save_checkpoint_step = 2500,
    max_steps = epochs,
    train_hooks = train_hooks,
    eval_step = 0,
)
