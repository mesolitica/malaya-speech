from scipy.signal import lfilter, butter
from scipy.io.wavfile import read
from scipy import interpolate
import numpy as np
import librosa
import decimal
import math
from python_speech_features import fbank


class SpeakerNetFeaturizer:
    def __init__(self, speech_config):
        self.sample_rate = speech_config['sample_rate']
        self.frame_length = int(
            self.sample_rate * (speech_config['frame_ms'] / 1000)
        )
        self.frame_step = int(
            self.sample_rate * (speech_config['stride_ms'] / 1000)
        )
        self.n_fft = speech_config['n_fft']
        self.num_feature_bins = speech_config['num_feature_bins']
        self.preemphasis = speech_config['preemphasis']
        self.normalize_signal = speech_config['normalize_signal']
        self.normalize_feature = speech_config['normalize_feature']
        self.normalize_per_feature = speech_config['normalize_per_feature']

        self.mel_basis = librosa.filters.mel(
            self.sample_rate,
            self.n_fft,
            n_mels = self.num_feature_bins,
            fmin = 0,
            fmax = self.sample_rate / 2,
        )

    def vectorize(self, signal):
        signal = preemphasis(signal, self.preemphasis)
        spect = np.abs(
            librosa.stft(
                signal,
                n_fft = self.n_fft,
                hop_length = self.frame_step,
                win_length = self.frame_length,
            )
        )
        spect = np.power(spect, 2)
        mel = np.matmul(self.mel_basis, spect)
        log_zero_guard_value = 2 ** -24
        features = np.log(mel + log_zero_guard_value)
        features = normalize_batch(np.expand_dims(features, 0))[0]
        return features.T

    def __call__(self, signal):
        return self.vectorize(signal)


def normalize_batch(x, CONSTANT = 1e-5):
    x_mean = np.zeros((1, x.shape[1]), dtype = x.dtype)
    x_std = np.zeros((1, x.shape[1]), dtype = x.dtype)
    x_mean[0, :] = x[0].mean(axis = 1)
    x_std[0, :] = x[0].std(axis = 1)
    x_std += CONSTANT
    return (x - np.expand_dims(x_mean, 2)) / np.expand_dims(x_std, 2)


def normalize(values):
    return (values - np.mean(values)) / np.std(values)


def power_spectrogram(
    audio_data,
    sample_rate = 16000,
    n_mels = 128,
    n_fft = 512,
    hop_length = 180,
    normalized = True,
):
    spectrogram = librosa.feature.melspectrogram(
        audio_data,
        sr = sample_rate,
        n_mels = n_mels,
        n_fft = n_fft,
        hop_length = hop_length,
    )

    log_spectrogram = librosa.power_to_db(spectrogram, ref = np.max)
    if normalized:
        log_spectrogram = normalize(log_spectrogram)

    v = log_spectrogram.T
    return v


# https://github.com/tensorflow/models/blob/master/research/deep_speech/data/featurizer.py#L24
def spectrogram(
    samples,
    sample_rate,
    stride_ms = 10.0,
    window_ms = 20.0,
    max_freq = None,
    eps = 1e-14,
    normalized = False,
):

    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError(
            'max_freq must not be greater than half of sample rate.'
        )

    if stride_ms > window_ms:
        raise ValueError('Stride size must not be greater than window size.')

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[: len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(
        samples, shape = nshape, strides = nstrides
    )
    assert np.all(
        windows[:, 1] == samples[stride_size : (stride_size + window_size)]
    )

    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis = 0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= 2.0 / scale
    fft[(0, -1), :] /= scale
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)

    specgram = np.transpose(specgram, (1, 0)).astype(np.float32)

    if normalized:
        specgram = normalize(specgram)

    return specgram


def mfcc_delta(signal, freq = 16000, n_mfcc = 42, size = 512, step = 16):
    # Mel Frequency Cepstral Coefficents
    mfcc = librosa.feature.mfcc(
        y = signal, sr = freq, n_mfcc = n_mfcc, n_fft = size, hop_length = step
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)

    # Root Mean Square Energy
    mel_spectogram = librosa.feature.melspectrogram(
        y = signal, sr = freq, n_fft = size, hop_length = step
    )
    rmse = librosa.feature.rms(
        S = mel_spectogram, frame_length = size, hop_length = step
    )

    mfcc = np.asarray(mfcc)
    mfcc_delta = np.asarray(mfcc_delta)
    mfcc_delta2 = np.asarray(mfcc_delta2)
    rmse = np.asarray(rmse)

    features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis = 0)
    return features.T


# for VGGVox v1
def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)
    end_frame = int(max_sec * frames_per_sec)
    step_frame = int(step_sec * frames_per_sec)
    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


# for VGGVox v1
def get_buckets(max_sec = 10, bucket_step = 1, frame_step = 0.01):
    return build_buckets(max_sec, bucket_step, frame_step)


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
# for VGGVox v1
def round_half_up(number):
    return int(
        decimal.Decimal(number).quantize(
            decimal.Decimal('1'), rounding = decimal.ROUND_HALF_UP
        )
    )


# for VGGVox v1
def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print('Sample rate must be 16kHz or 8kHz only')
        exit(1)
    sin = lfilter([1, -1], [1, -alpha], sin)
    dither = (
        np.random.random_sample(len(sin))
        + np.random.random_sample(len(sin))
        - 1
    )
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout


# for VGGVox v1
def preemphasis(signal, coeff = 0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# for VGGVox v1
def rolling_window(a, window, step = 1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape = shape, strides = strides)[
        ::step
    ]


# for VGGVox v1
def framesig(
    sig,
    frame_len,
    frame_step,
    winfunc = lambda x: numpy.ones((x,)),
    stride_trick = True,
):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(
            math.ceil((1.0 * slen - frame_len) / frame_step)
        )  # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(
            padsignal, window = frame_len, step = frame_step
        )
    else:
        indices = (
            numpy.tile(numpy.arange(0, frame_len), (numframes, 1))
            + numpy.tile(
                numpy.arange(0, numframes * frame_step, frame_step),
                (frame_len, 1),
            ).T
        )
        indices = numpy.array(indices, dtype = numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


# for VGGVox v1
def normalize_frames(m, epsilon = 1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


# for VGGVox v1
def vggvox_v1(
    signal,
    sample_rate = 16000,
    preemphasis_alpha = 0.97,
    frame_len = 0.025,
    frame_step = 0.01,
    num_fft = 512,
    buckets = None,
    minlen = 100,
    **kwargs,
):
    signal = signal.copy()
    signal *= 2 ** 15
    signal = remove_dc_and_dither(signal, sample_rate)
    signal = preemphasis(signal, coeff = preemphasis_alpha)
    frames = framesig(
        signal,
        frame_len = frame_len * sample_rate,
        frame_step = frame_step * sample_rate,
        winfunc = np.hamming,
    )
    fft = abs(np.fft.fft(frames, n = num_fft))
    fft_norm = normalize_frames(fft.T)

    if buckets:
        rsize = max(k for k in buckets if k <= fft_norm.shape[1])
        rstart = int((fft_norm.shape[1] - rsize) / 2)
        out = fft_norm[:, rstart : rstart + rsize]
        return out

    else:
        if fft_norm.shape[1] < minlen:
            fft_norm = np.pad(
                fft_norm, ((0, 0), (0, minlen - fft_norm.shape[1])), 'constant'
            )
        return fft_norm.astype('float32')


# for VGGVox v2
def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft = 1024):
    linear = librosa.stft(
        wav, n_fft = n_fft, win_length = win_length, hop_length = hop_length
    )  # linear spectrogram
    return linear.T


def vggvox_v2(
    signal,
    win_length = 400,
    sr = 16000,
    hop_length = 160,
    n_fft = 512,
    spec_len = 100,
    mode = 'train',
    concat = True,
    **kwargs,
):
    if concat:
        wav = np.append(signal, signal[::-1])
    else:
        wav = signal
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time < spec_len:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
        else:
            spec_mag = mag_T
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims = True)
    std = np.std(spec_mag, 0, keepdims = True)
    return (spec_mag - mu) / (std + 1e-5)


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
        mel = factor * np.log10(mel + 1e-8)
        mel = np.clip((mel - ref_db + max_db) / max_db, 1e-11, 1)
    return mel


def unscale_mel(mel, ref_db = 20, max_db = 100, factor = 15):
    inv_mel = ((mel * max_db) - max_db + ref_db) / factor
    inv_mel = np.power(10, inv_mel) - 1e-8
    inv_mel[inv_mel < 5e-6] = 0.0
    return inv_mel


def scale_mel_vocoder(
    y,
    sr = 22050,
    n_fft = 1024,
    hop_length = 256,
    win_length = None,
    n_mels = 256,
    ref_db = 20,
    max_db = 200,
    factor = 13,
    fmin = 80,
    fmax = 7600,
    scale = True,
):
    D = librosa.stft(
        y,
        n_fft = 1024,
        hop_length = 256,
        win_length = None,
        window = 'hann',
        pad_mode = 'reflect',
    )
    S, _ = librosa.magphase(D)
    mel_basis = librosa.filters.mel(
        sr = sr, n_fft = n_fft, n_mels = n_mels, fmin = fmin, fmax = fmax
    )
    mel = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10))
    if scale:
        mel = np.clip(((factor * mel) - ref_db + max_db) / max_db, 1e-11, 1)
    return mel


def unscale_mel_vocoder(mel, ref_db = 20, max_db = 200, factor = 13):
    inv_mel = ((mel * max_db) - max_db + ref_db) / factor
    return inv_mel


def mu_law(x, mu = 255, int8 = False):
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    out = np.floor(out * 128)
    if int8:
        out = out.astype(np.int8)
    return out


def inverse_mu_law(x, mu = 255.0):
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2.0 / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu) ** np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out


def mfcc_fbank(signal, sample_rate):
    filter_banks, energies = fbank(signal, samplerate = sample_rate, nfilt = 64)
    frames_features = normalize_frames(filter_banks)
    return np.array(frames_features, dtype = np.float32)


def read_mfcc(signal, sr = 16000, voice_only = True, **kwargs):
    if voice_only:
        energy = np.abs(signal)
        silence_threshold = np.percentile(energy, 95)
        offsets = np.where(energy > silence_threshold)[0]
        audio_voice_only = signal[offsets[0] : offsets[-1]]
    else:
        audio_voice_only = signal
    mfcc = mfcc_fbank(audio_voice_only, sr)
    return mfcc
