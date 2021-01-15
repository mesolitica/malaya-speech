import numpy as np
import soundfile as sf
from scipy.io.wavfile import read
from scipy import interpolate
from herpetologist import check_type


def resample(data, old_samplerate, new_samplerate):
    """
    Resample signal.

    Parameters
    ----------
    data: np.array
    old_samplerate: int
        old sample rate.
    new_samplerate: int
        new sample rate.

    Returns
    -------
    result: data
    """
    old_audio = data
    duration = data.shape[0] / old_samplerate
    time_old = np.linspace(0, duration, old_audio.shape[0])
    time_new = np.linspace(
        0, duration, int(old_audio.shape[0] * new_samplerate / old_samplerate)
    )

    interpolator = interpolate.interp1d(time_old, old_audio.T)
    data = interpolator(time_new).T
    return data


def read_audio(data, old_samplerate, sample_rate = 22050):
    if len(data.shape) == 2:
        data = data[:, 0]

    if old_samplerate != sample_rate and sample_rate is not None:
        data = resample(data, old_samplerate, sample_rate)
    else:
        sample_rate = old_samplerate

    return data, sample_rate


@check_type
def load(file: str, sr = 16000, scale: bool = True):
    """
    Read sound file, any format supported by soundfile.read

    Parameters
    ----------
    file: str
    sr: int, (default=16000)
        new sample rate. If input sample rate is not same, will resample automatically.
    scale: bool, (default=True)
        Scale to -1 and 1.

    Returns
    -------
    result: (y, sr)
    """
    data, old_samplerate = sf.read(file)
    y, sr = read_audio(data, old_samplerate, sr)
    if scale:
        y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr
