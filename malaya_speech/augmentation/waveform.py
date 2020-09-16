import numpy as np
import librosa
import os
import scipy
import json


def random_pitch(samples, low = 0.5, high = 1.0):
    y_pitch_speed = samples.copy()
    length_change = np.random.uniform(low = low, high = high)
    speed_fac = 1.0 / length_change
    tmp = np.interp(
        np.arange(0, len(y_pitch_speed), speed_fac),
        np.arange(0, len(y_pitch_speed)),
        y_pitch_speed,
    )
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[:minlen] = tmp[:minlen]
    return y_pitch_speed


def random_amplitude(samples, low = 1.5, high = 3):
    y_aug = samples.copy()
    dyn_change = np.random.uniform(low = low, high = high)
    return y_aug * dyn_change


def random_stretch(samples, low = 0.5, high = 1.3):
    input_length = len(samples)
    stretching = samples.copy()
    random_stretch = np.random.uniform(low = low, high = high)
    stretching = librosa.effects.time_stretch(
        stretching.astype('float'), random_stretch
    )
    return stretching


def add_uniform_noise(samples):
    y_noise = samples.copy()
    noise_amp = 0.01 * np.random.uniform() * np.amax(y_noise)
    return y_noise.astype('float64') + noise_amp * np.random.normal(
        size = y_noise.shape[0]
    )


def add_noise(samples, noise, random_sample = True, factor = 0.1):
    y_noise = samples.copy()
    if len(y_noise) > len(noise):
        noise = np.tile(noise, int(np.ceil(len(y_noise) / len(noise))))
    else:
        if random_sample:
            noise = noise[np.random.randint(0, len(noise) - len(y_noise) + 1) :]
    return y_noise + noise[: len(y_noise)] * factor


# from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def fftnoise(f):
    f = np.array(f, dtype = 'complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def add_band_limited_noise(
    samples,
    min_freq = 2000,
    max_freq = 12000,
    sample_size = 1024,
    samplerate = 1,
):
    y_noise = samples.copy()
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(sample_size)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    noise = fftnoise(f)
    return add_noise(samples, noise, factor = 1.0)
