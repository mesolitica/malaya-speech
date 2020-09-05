import numpy as np
import soundfile as sf
from scipy.io.wavfile import read
from scipy.signal import resample
from scipy import interpolate
from herpetologist import check_type
from malaya_speech.utils.astype import float_to_int


def change_samplerate(data, old_samplerate, new_samplerate):
    old_audio = data
    duration = data.shape[0] / old_samplerate
    time_old = np.linspace(0, duration, old_audio.shape[0])
    time_new = np.linspace(
        0, duration, int(old_audio.shape[0] * new_samplerate / old_samplerate)
    )

    interpolator = interpolate.interp1d(time_old, old_audio.T)
    data = interpolator(time_new).T
    return data


def read_audio(data, old_samplerate, sample_rate = 16000):
    if len(data.shape) == 2:
        data = data[:, 0]

    if old_samplerate != sample_rate:
        data = change_samplerate(data, old_samplerate, sample_rate)

    return data.tolist(), sample_rate


@check_type
def wav(file: str, sample_rate: int = 16000):
    """
    Read wav file.

    Parameters
    ----------
    file: str
    sample_rate: int, (default=16000)
        new sample rate. If input sample rate is not same, will interpolate automatically.

    Returns
    -------
    result: (y, new_sample_rate)
    """

    old_samplerate, data = read(file)
    y, sr = read_audio(data, old_samplerate, sample_rate)
    return float_to_int(y), sr


@check_type
def flac(file: str, sample_rate: int = 16000):
    """
    Read flac file.

    Parameters
    ----------
    file: str
    sample_rate: int, (default=16000)
        new sample rate. If input sample rate is not same, will interpolate automatically.

    Returns
    -------
    result: (y, new_sample_rate)
    """

    data, old_samplerate = sf.read(file)
    y, sr = read_audio(data, old_samplerate, sample_rate)
    return float_to_int(y), sr
