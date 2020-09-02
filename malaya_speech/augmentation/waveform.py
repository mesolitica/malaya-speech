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
    return stretch


def add_uniform_noise(samples):
    y_noise = samples.copy()
    noise_amp = 0.01 * np.random.uniform() * np.amax(y_noise)
    return y_noise.astype('float64') + noise_amp * np.random.normal(
        size = y_noise.shape[0]
    )


def add_noise(sample, noise, factor = 0.05):
    y_noise = samples.copy()
