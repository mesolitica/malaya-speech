import tensorflow as tf


class FDomainHelper(tf.keras.layers.Layer):
    def __init__(
        self,
        window_size=2048,
        hop_size=441,
        subband=None,
        **kwargs,
    ):
        super(FDomainHelper, self).__init__(**kwargs)
        self.subband = subband
        if self.subband is None:
            self.n_fft = window_size
            self.hop_size = hop_size
        else:
            self.n_fft = window_size // self.subband
            self.hop_size = hop_size // self.subband

        self.p = int((self.n_fft-self.hop_size)/2)

    def complex_spectrogram(self, input, eps=0.0):
        # [batchsize, samples]
        # return [batchsize, 2, t-steps, f-bins]

        padded = tf.pad(input, [[0, 0], [self.p, self.p]], mode='reflect')

        ffted_tf = tf.signal.stft(
            padded,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )  # [BC, T, n_fft/2+1]
        ffted_tf = tf.stack([tf.math.real(ffted_tf), tf.math.imag(ffted_tf)], axis=1)
        return ffted_tf

    def reverse_complex_spectrogram(self, input, eps=0.0, length=None):
        # [batchsize, 2[real,imag], t-steps, f-bins]

        ffted_tf = tf.complex(input[:, 0, :, :], input[:, 1, :, :])
        output = tf.signal.inverse_stft(
            ffted_tf,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
        )[:, self.p:-self.p]
        return output

    def spectrogram(self, input, eps=0.0):

        padded = tf.pad(input, [[0, 0], [self.p, self.p]], mode='reflect')

        ffted_tf = tf.signal.stft(
            padded,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )  # [BC, T, n_fft/2+1]

        real, imag = tf.math.real(ffted_tf), tf.math.imag(ffted_tf)
        added = real**2 + imag**2
        return tf.clip_by_value(added, eps, tf.reduce_max(added)) ** 0.5

    def spectrogram_phase(self, input, eps=0.0):
        padded = tf.pad(input, [[0, 0], [self.p, self.p]], mode='reflect')

        ffted_tf = tf.signal.stft(
            padded,
            self.n_fft,
            self.hop_size,
            fft_length=None,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )  # [BC, T, n_fft/2+1]

        real, imag = tf.math.real(ffted_tf), tf.math.imag(ffted_tf)
        added = real**2 + imag**2
        mag = tf.clip_by_value(added, eps, tf.reduce_max(added)) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin
