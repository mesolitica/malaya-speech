import numpy as np
import librosa
import os
import scipy
import json
from scipy.special import expit


def sox_reverb(
    y, reverberance = 1, hf_damping = 1, room_scale = 1, stereo_depth = 1
):
    from pysndfx import AudioEffectsChain

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
    from pysndfx import AudioEffectsChain

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
    from pysndfx import AudioEffectsChain

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
    from pysndfx import AudioEffectsChain

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


def random_pitch(sample, low = 0.5, high = 1.0):
    y_pitch_speed = sample.copy()
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


def random_amplitude(sample, low = 3, high = 5):
    y_aug = sample.copy()
    dyn_change = np.random.uniform(low = low, high = high)
    y_aug = y_aug * dyn_change
    return np.clip(y_aug, -1, 1)


def random_amplitude_threshold(sample, low = 3, high = 5, threshold = 0.4):
    y_aug = sample.copy()
    dyn_change = np.random.uniform(low = low, high = high)
    y_aug[np.abs(y_aug) >= threshold] = (
        y_aug[np.abs(y_aug) >= threshold] * dyn_change
    )
    return np.clip(y_aug, -1, 1)


def random_stretch(sample, low = 0.5, high = 1.3):
    input_length = len(sample)
    stretching = sample.copy()
    random_stretch = np.random.uniform(low = low, high = high)
    stretching = librosa.effects.time_stretch(
        stretching.astype('float'), random_stretch
    )
    return stretching


def random_sampling(sample, sr, length = 500):
    sr = int(sr / 1000)
    up = len(sample) - (sr * length)
    if up < 1:
        r = 0
    else:
        r = np.random.randint(0, up)
    return sample[r : r + sr * length]


def add_uniform_noise(sample, power = 0.01, return_noise = False):
    y_noise = sample.copy()
    noise_amp = power * np.random.uniform() * np.amax(y_noise)
    noise = noise_amp * np.random.normal(size = y_noise.shape[0])
    y_noise = y_noise + noise
    noise = noise / np.max(np.abs(y_noise))
    y_noise = y_noise / np.max(np.abs(y_noise))
    if return_noise:
        return y_noise, noise
    else:
        return y_noise


def add_noise(
    sample, noise, random_sample = True, factor = 0.1, return_noise = False
):
    y_noise = sample.copy()
    if len(y_noise) > len(noise):
        noise = np.tile(noise, int(np.ceil(len(y_noise) / len(noise))))
    else:
        if random_sample:
            noise = noise[np.random.randint(0, len(noise) - len(y_noise) + 1) :]
    noise = noise[: len(y_noise)] * factor
    y_noise = y_noise + noise
    noise = noise / np.max(np.abs(y_noise))
    y_noise = y_noise / np.max(np.abs(y_noise))
    if return_noise:
        return y_noise, noise
    else:
        return y_noise


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
    sample, sr = 16000, min_freq = 200, max_freq = 24000
):
    freqs = np.abs(np.fft.fftfreq(len(sample), 1 / sr))
    f = np.zeros(len(sample))
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return sample + fftnoise(f)
