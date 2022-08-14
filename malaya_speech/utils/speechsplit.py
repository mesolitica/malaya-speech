import numpy as np
from malaya_speech.utils.importer import pyworld_exist, pysptk_exist, pw, sptk

A = np.array(
    [1.0, -4.97233633, 9.88972758, -9.83516149, 4.89048558, -0.97271534]
)
B = np.array(
    [0.98626332, -4.93131661, 9.86263322, -9.86263322, 4.93131661, -0.98626332]
)
sr = 22050


def quantize_f0_numpy(x, num_bins=256):
    # x is logf0
    assert x.ndim == 1
    x = x.astype(float).copy()
    uv = x <= 0
    x[uv] = 0.0
    x = np.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins + 1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc, x.astype(np.int64)


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    f0 = f0.astype(float).copy()
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


def get_f0_sptk(wav, lo, hi):

    if not pysptk_exist:
        raise Exception('pysptk not installed. Please install it by `pip3 install pysptk` and try again.')

    f0_rapt = sptk.rapt(
        wav.astype(np.float32) * 32768, sr, 256, min=lo, max=hi, otype=2
    )
    index_nonzero = f0_rapt != -1e10
    mean_f0, std_f0 = (
        np.mean(f0_rapt[index_nonzero]),
        np.std(f0_rapt[index_nonzero]),
    )
    return speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)


def get_f0_pyworld(wav):

    if not pyworld_exist:
        raise Exception('pyworld not installed. Please install it by `pip3 install pyworld` and try again.')

    _f0, t = pw.dio(wav, sr, f0_ceil=7600, frame_period=1000 * 256 / sr)
    f0_rapt = pw.stonemask(wav.astype(np.double), _f0, t, sr)
    index_nonzero = f0_rapt != 0.0
    mean_f0, std_f0 = (
        np.mean(f0_rapt[index_nonzero]),
        np.std(f0_rapt[index_nonzero]),
    )
    return speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
