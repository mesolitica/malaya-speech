import librosa
import numpy as np
import python_speech_features
from pysndfx import AudioEffectsChain
from malaya_speech.utils.astype import int_to_float

# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_power(y, sr = 16000):
    y = int_to_float(y)
    cent = librosa.feature.spectral_centroid(y = y, sr = sr)

    threshold_h = round(np.median(cent)) * 1.5
    threshold_l = round(np.median(cent)) * 0.1

    less_noise = (
        AudioEffectsChain()
        .lowshelf(gain = -30.0, frequency = threshold_l, slope = 0.8)
        .highshelf(gain = -12.0, frequency = threshold_h, slope = 0.5)
    )
    y_clean = less_noise(y)

    return y_clean


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_centroid_s(y, sr = 16000):
    y = int_to_float(y)
    cent = librosa.feature.spectral_centroid(y = y, sr = sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = (
        AudioEffectsChain()
        .lowshelf(gain = -12.0, frequency = threshold_l, slope = 0.5)
        .highshelf(gain = -12.0, frequency = threshold_h, slope = 0.5)
        .limiter(gain = 6.0)
    )

    y_cleaned = less_noise(y)

    return y_cleaned


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_centroid_mb(y, sr = 16000):
    y = int_to_float(y)
    cent = librosa.feature.spectral_centroid(y = y, sr = sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = (
        AudioEffectsChain()
        .lowshelf(gain = -30.0, frequency = threshold_l, slope = 0.5)
        .highshelf(gain = -30.0, frequency = threshold_h, slope = 0.5)
        .limiter(gain = 10.0)
    )
    y_cleaned = less_noise(y)

    cent_cleaned = librosa.feature.spectral_centroid(y = y_cleaned, sr = sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows / 3 * 2)
    boost_l = math.floor(rows / 6)
    boost = math.floor(rows / 3)
    boost_bass = AudioEffectsChain().lowshelf(
        gain = 16.0, frequency = boost_h, slope = 0.5
    )
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_mfcc_down(y, sr = 16000):
    y = int_to_float(y)
    hop_length = 512

    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n ** 2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = (
        AudioEffectsChain()
        .highshelf(frequency = min_hz * (-1) * 1.2, gain = -12.0, slope = 0.6)
        .limiter(gain = 8.0)
    )
    y_speach_boosted = speech_booster(y)

    return y_speach_boosted


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_mfcc_up(y, sr = 16000):
    y = int_to_float(y)
    hop_length = 512
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n ** 2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(
        frequency = min_hz * (-1), gain = 12.0, slope = 0.5
    )
    y_speach_boosted = speech_booster(y)

    return y_speach_boosted


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_median(y, sr = 16000):
    y = int_to_float(y)
    y = sp.signal.medfilt(y, 3)
    return y


def trim_silence(y, return_trimmed_length = False):
    y = int_to_float(y)
    y_trimmed, index = librosa.effects.trim(
        y, top_db = 20, frame_length = 2, hop_length = 500
    )
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)

    if return_trimmed_length:
        return y_trimmed, trimmed_length
    else:
        return y_trimmed
