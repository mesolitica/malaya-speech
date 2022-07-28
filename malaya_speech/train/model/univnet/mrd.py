import tensorflow as tf
from ..melgan.layer import WeightNormalization
from ..utils import shape_list


class DiscriminatorR(tf.keras.layers.Layer):
    def __init__(self, hp, resolution, **kwargs):
        super(DiscriminatorR, self).__init__(**kwargs)

        self.resolution = resolution
        self.LRELU_SLOPE = hp.mpd.lReLU_slope

        norm_f = WeightNormalization

        self.convs = [
            norm_f(tf.keras.layers.Conv2D(32, (3, 9), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(32, (3, 9), (1, 2), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(32, (3, 9), (1, 2), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(32, (3, 9), (1, 2), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(32, (3, 3), padding='SAME')),
        ]
        self.conv_post = norm_f(tf.keras.layers.Conv2D(1, (3, 3), 1, padding='SAME'))

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x_ = tf.pad(x, [[0, 0], [int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)], [0, 0]], mode='REFLECT')
        x_ = x_[:, :, 0]
        x = tf.abs(
            tf.signal.stft(
                signals=x_,
                frame_length=win_length,
                frame_step=hop_length,
                fft_length=n_fft,
            )
        )
        x = tf.expand_dims(x, -1)

        return x

    def call(self, x):
        fmap = []
        b, t, c = shape_list(x)
        x = self.spectrogram(x)

        for l in self.convs:
            x = l(x)
            x = tf.keras.layers.LeakyReLU(alpha=self.LRELU_SLOPE)(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = tf.reshape(x, [b, -1])

        return fmap, x


class MultiResolutionDiscriminator(tf.keras.layers.Layer):
    def __init__(self, hp, **kwargs):
        super(MultiResolutionDiscriminator, self).__init__(**kwargs)

        self.resolutions = eval(hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(hp, resolution) for resolution in self.resolutions]

    def call(self, y):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(y))
        return ret
