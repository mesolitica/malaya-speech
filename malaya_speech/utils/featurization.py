from scipy.signal import lfilter, butter
from scipy.io.wavfile import read
from scipy import interpolate
import numpy as np
import librosa
import decimal
import math
from python_speech_features import fbank


class SpeakerNetFeaturizer:
    def __init__(
        self,
        sample_rate = 16000,
        frame_ms = 20,
        stride_ms = 10,
        n_fft = 512,
        num_feature_bins = 64,
        preemphasis = 0.97,
        normalize_signal = True,
        normalize_feature = True,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(self.sample_rate * (frame_ms / 1000))
        self.frame_step = int(self.sample_rate * (stride_ms / 1000))
        self.n_fft = n_fft
        self.num_feature_bins = num_feature_bins
        self.preemphasis = preemphasis
        self.normalize_signal = normalize_signal
        self.normalize_feature = normalize_feature

        self.mel_basis = librosa.filters.mel(
            self.sample_rate,
            self.n_fft,
            n_mels = self.num_feature_bins,
            fmin = 0,
            fmax = self.sample_rate / 2,
        )

    def vectorize(self, signal):
        if self.normalize_signal:
            signal = normalize_signal(signal)

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
        if self.normalize_feature:
            features = normalize_batch(np.expand_dims(features, 0))[0]
        return features.T

    def __call__(self, signal):
        return self.vectorize(signal)


class STTFeaturizer:
    def __init__(
        self,
        sample_rate = 16000,
        frame_ms = 25,
        stride_ms = 10,
        nfft = None,
        num_feature_bins = 80,
        feature_type = 'log_mel_spectrogram',
        preemphasis = 0.97,
        dither = 1e-5,
        normalize_signal = True,
        normalize_feature = True,
        norm_per_feature = True,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(self.sample_rate * (frame_ms / 1000))
        self.frame_step = int(self.sample_rate * (stride_ms / 1000))
        self.num_feature_bins = num_feature_bins
        self.feature_type = feature_type
        self.preemphasis = preemphasis
        self.dither = dither
        self.normalize_signal = normalize_signal
        self.normalize_feature = normalize_feature
        self.norm_per_feature = norm_per_feature
        self.nfft = nfft or 2 ** math.ceil(
            math.log2((frame_ms / 1000) * self.sample_rate)
        )
        self.window_fn = np.hanning

        self.mel_basis = librosa.filters.mel(
            self.sample_rate,
            self.nfft,
            n_mels = self.num_feature_bins,
            fmin = 0,
            fmax = int(self.sample_rate / 2),
        )

    def __call__(self, signal):
        return self.vectorize(signal)

    def vectorize(self, signal):
        if self.normalize_signal:
            signal = normalize_signal(signal)

        if self.dither > 0:
            signal += self.dither * np.random.randn(*signal.shape)

        signal = preemphasis(signal, coeff = self.preemphasis)

        if self.feature_type == 'spectrogram':
            powspec = np.square(
                np.abs(
                    librosa.core.stft(
                        signal,
                        n_fft = self.frame_length,
                        hop_length = self.frame_step,
                        win_length = self.frame_length,
                        center = True,
                        window = window_fn,
                    )
                )
            )
            powspec[powspec <= 1e-30] = 1e-30
            features = 10 * np.log10(powspec.T)
            features = features[:, :num_features]

        elif self.feature_type == 'mfcc':
            S = np.square(
                np.abs(
                    librosa.core.stft(
                        signal,
                        n_fft = self.nfft,
                        hop_length = self.frame_step,
                        win_length = self.frame_length,
                        center = True,
                        window = self.window_fn,
                    )
                )
            )
            features = librosa.feature.mfcc(
                sr = self.sample_rate,
                S = S,
                n_mfcc = self.num_feature_bins,
                n_mels = 2 * self.num_feature_bins,
            ).T

        elif self.feature_type == 'log_mel_spectrogram':
            S = (
                np.abs(
                    librosa.core.stft(
                        signal,
                        n_fft = self.nfft,
                        hop_length = self.frame_step,
                        win_length = self.frame_length,
                        center = True,
                        window = self.window_fn,
                    )
                )
                ** 2.0
            )
            features = np.log(np.dot(self.mel_basis, S) + 1e-20).T

        else:
            raise ValueError(
                "feature_type must be either 'mfcc', "
                "'log_mel_spectrogram', or 'spectrogram' "
            )

        if self.normalize_feature:
            norm_axis = 0 if self.norm_per_feature else None
            mean = np.mean(features, axis = norm_axis)
            std_dev = np.std(features, axis = norm_axis)
            features = (features - mean) / std_dev

        return features


def normalize_signal(signal, gain = None):
    """
    Normalize float32 signal to [-1, 1] range
    """
    if gain is None:
        gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain


def normalize_batch(x, CONSTANT = 1e-5):
    x_mean = np.zeros((1, x.shape[1]), dtype = x.dtype)
    x_std = np.zeros((1, x.shape[1]), dtype = x.dtype)
    x_mean[0, :] = x[0].mean(axis = 1)
    x_std[0, :] = x[0].std(axis = 1)
    x_std += CONSTANT
    return (x - np.expand_dims(x_mean, 2)) / np.expand_dims(x_std, 2)


def normalize(values, CONSTANT = 0):
    return (values - np.mean(values)) / (np.std(values) + CONSTANT)


def preemphasis(signal, coeff = 0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def normalize_frames(m, epsilon = 1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


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

    linear = librosa.stft(
        wav, n_fft = n_fft, win_length = win_length, hop_length = hop_length
    )
    linear_spect = linear.T
    mag, _ = librosa.magphase(linear_spect)
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time < spec_len:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
        else:
            spec_mag = mag_T
    else:
        spec_mag = mag_T

    mu = np.mean(spec_mag, 0, keepdims = True)
    std = np.std(spec_mag, 0, keepdims = True)
    return (spec_mag - mu) / (std + 1e-5)


def deep_speaker(signal, sr = 16000, voice_only = True, **kwargs):
    if voice_only:
        energy = np.abs(signal)
        silence_threshold = np.percentile(energy, 95)
        offsets = np.where(energy > silence_threshold)[0]
        audio_voice_only = signal[offsets[0] : offsets[-1]]
    else:
        audio_voice_only = signal
    filter_banks, energies = fbank(signal, samplerate = sr, nfilt = 64)
    frames_features = normalize_frames(filter_banks)
    mfcc = np.array(frames_features, dtype = np.float32)
    return mfcc


def to_mel(
    signal,
    sampling_rate = 22050,
    fft_size = 1024,
    hop_size = 256,
    win_length = None,
    window = 'hann',
    fmin = 80,
    fmax = 7600,
    trim_threshold_in_db = 60,
    trim_frame_size = 2048,
    trim_hop_size = 512,
    trim_silence = True,
):
    if trim_silence:
        signal, _ = librosa.effects.trim(
            signal,
            top_db = trim_threshold_in_db,
            frame_length = trim_frame_size,
            hop_length = trim_hop_size,
        )
    D = librosa.stft(
        signal,
        n_fft = fft_size,
        hop_length = hop_size,
        win_length = win_length,
        window = window,
        pad_mode = 'reflect',
    )
    S, _ = librosa.magphase(D)
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate // 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr = sampling_rate,
        n_fft = fft_size,
        n_mels = num_mels,
        fmin = fmin,
        fmax = fmax,
    )
    mel = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)).T
    signal = np.pad(signal, (0, fft_size), mode = 'edge')
    signal = signal[: len(mel) * hop_size]
    return signal
