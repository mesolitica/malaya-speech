import tensorflow as tf
from ..melgan.layer import WeightNormalization
from ..utils import shape_list


class DiscriminatorP(tf.keras.layers.Layer):
    def __init__(self, hp, period, **kwargs):
        super(DiscriminatorP, self).__init__(**kwargs)

        self.LRELU_SLOPE = hp.mpd.lReLU_slope
        self.period = period

        kernel_size = hp.mpd.kernel_size
        stride = hp.mpd.stride

        norm_f = WeightNormalization

        self.convs = [
            norm_f(tf.keras.layers.Conv2D(64, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(128, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(256, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(512, (kernel_size, 1), (stride, 1), padding='SAME')),
            norm_f(tf.keras.layers.Conv2D(1024, (kernel_size, 1), 1, padding='SAME')),
        ]
        self.conv_post = norm_f(tf.keras.layers.Conv2D(1, (3, 1), 1, padding='SAME'))

    def call(self, x):
        fmap = []
        b, t, c = shape_list(x)

        def f1():
            n_pad = self.period - (t % self.period)
            x_ = tf.pad(x, [[0, 0], [0, n_pad], [0, 0]], mode='REFLECT')
            return x_

        x = tf.cond(tf.math.not_equal(t % self.period, 0), f1, lambda: x)
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t // self.period, self.period, c])

        for l in self.convs:
            x = l(x)
            x = tf.keras.layers.LeakyReLU(alpha=self.LRELU_SLOPE)(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        x = tf.reshape(x, [b, -1])

        return fmap, x


class MultiPeriodDiscriminator(tf.keras.layers.Layer):
    def __init__(self, hp, **kwargs):
        super(MultiPeriodDiscriminator, self).__init__(**kwargs)

        discs = [DiscriminatorP(hp, period) for period in hp.mpd.periods]
        self.discriminators = discs

    def call(self, y):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(y))
        return ret
