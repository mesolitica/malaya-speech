from scipy.io.wavfile import read
from scipy.signal import resample
from scipy import interpolate


class AudioEncoder(object):
    def __init__(self, num_reserved_ids = 0, sample_rate = 16000):
        assert num_reserved_ids == 0
        self._sample_rate = sample_rate

    def encode(self, s):

        old_samplerate, data = read(s)
        if len(data.shape) == 2:
            data = data[:, 0]

        if data.dtype not in [np.float32, np.float64]:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        if old_samplerate != self._sample_rate:
            old_audio = data
            duration = data.shape[0] / old_samplerate

            time_old = np.linspace(0, duration, old_audio.shape[0])
            time_new = np.linspace(
                0,
                duration,
                int(old_audio.shape[0] * self._sample_rate / old_samplerate),
            )

            interpolator = interpolate.interp1d(time_old, old_audio.T)
            data = interpolator(time_new).T

        return data.tolist(), data.shape[0] / self._sample_rate


def normalize(values):
    """
    Normalize values to mean 0 and std 1
    """
    return (values - np.mean(values)) / np.std(values)


def calc_power_spectrogram(
    audio_data, samplerate = 22050, n_mels = 128, n_fft = 512, hop_length = 180
):
    """
    Calculate power spectrogram from the given raw audio data
    Args:
    audio_data: numpyarray of raw audio wave
    samplerate: the sample rate of the `audio_data`
    n_mels: the number of mels to generate
    n_fft: the window size of the fft
    hop_length: the hop length for the window
    Returns: the spectrogram in the form [time, n_mels]
    """
    spectrogram = librosa.feature.melspectrogram(
        audio_data,
        sr = samplerate,
        n_mels = n_mels,
        n_fft = n_fft,
        hop_length = hop_length,
    )

    # convert to log scale (dB)
    log_spectrogram = librosa.power_to_db(spectrogram, ref = np.max)

    # normalize
    normalized_spectrogram = normalize(log_spectrogram)

    v = normalized_spectrogram.T
    return v
