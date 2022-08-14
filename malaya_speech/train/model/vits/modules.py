import tensorflow.compat.v1 as tf
import math
from ..fastspeech.layer import gelu
from ..melgan.layer import WeightNormalization
from ..utils import WeightNormalization as K_WeightNormalization
from ..utils import shape_list
from . import commons
from .transforms import piecewise_rational_quadratic_transform_tf

LRELU_SLOPE = 0.1


class ConvReluNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_channels, out_channels, kernel_size, n_layers, p_dropout, **kwargs):
        super(ConvReluNorm, self).__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = []
        self.norm_layers = []

        self.conv_layers.append(tf.keras.layers.Conv1D(hidden_channels, kernel_size, padding='SAME'))
        self.norm_layers.append(tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5))
        self.relu_drop = tf.keras.Sequential([tf.keras.layers.ReLU(),
                                              tf.keras.layers.Dropout(p_dropout)])
        for _ in range(n_layers-1):
            self.conv_layers.append(tf.keras.layers.Conv1D(hidden_channels, kernel_size, padding='SAME'))
            self.norm_layers.append(tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5))

        self.proj = tf.keras.layers.Conv1D(out_channels, 1, padding='SAME',
                                           kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, x, x_mask, training=True):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x, training=training)
            x = self.relu_drop(x, training=training)
        x = x_org + self.proj(x)
        return x * x_mask


class WN(tf.keras.layers.Layer):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, **kwargs):
        super(WN, self).__init__(**kwargs)

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = []
        self.res_skip_layers = []
        self.drop = tf.keras.layers.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = tf.keras.layers.Conv1D(2*hidden_channels*n_layers, 1, padding='SAME')
            self.cond_layer = WeightNormalization(cond_layer)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            in_layer = tf.keras.layers.Conv1D(2*hidden_channels, kernel_size,
                                              dilation_rate=dilation, padding='SAME')
            in_layer = WeightNormalization(in_layer)
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = tf.keras.layers.Conv1D(res_skip_channels, 1, padding='SAME')
            res_skip_layer = WeightNormalization(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def call(self, x, x_mask, g=None, training=True):
        output = tf.zeros_like(x)
        n_channels_tensor = self.hidden_channels

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, :, cond_offset:cond_offset+2*self.hidden_channels]
            else:
                g_l = tf.zeros_like(x_in)
            acts = commons.fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts, training=training)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :, :self.hidden_channels]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, :, self.hidden_channels:]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        try:
            if self.gin_channels != 0:
                self.gin_channels = self.gin_channels.remove()
            for i in range(len(self.in_layers)):
                self.in_layers[i] = self.in_layers[i].remove()
            for i in range(len(self.res_skip_layers)):
                self.res_skip_layers[i] = self.res_skip_layers[i].remove()
        except Exception as e:
            print(e)


class ResBlock1(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), **kwargs):
        super(ResBlock1, self).__init__(**kwargs)
        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.01, seed=None
        )
        self.convs1 = [
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=dilation[0],
                                                       padding='SAME', kernel_initializer=initializer)),
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=dilation[1],
                                                       padding='SAME', kernel_initializer=initializer)),
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=dilation[2],
                                                       padding='SAME', kernel_initializer=initializer))
        ]
        self.convs2 = [
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=1,
                                                       padding='SAME', kernel_initializer=initializer)),
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=1,
                                                       padding='SAME', kernel_initializer=initializer)),
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=1,
                                                       padding='SAME', kernel_initializer=initializer))
        ]

    def call(self, x, x_mask=None, training=True):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = tf.keras.layers.LeakyReLU(alpha=LRELU_SLOPE)(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = tf.keras.layers.LeakyReLU(alpha=LRELU_SLOPE)(xt)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class ResBlock2(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3), **kwargs):
        super(ResBlock1, self).__init__(**kwargs)
        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.01, seed=None
        )
        self.convs = [
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=dilation[0],
                                                       padding='SAME', kernel_initializer=initializer)),
            WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=dilation[1],
                                                       padding='SAME', kernel_initializer=initializer)),
        ]

    def call(self, x, x_mask=None, training=True):
        for c in self.convs:
            xt = tf.keras.layers.LeakyReLU(alpha=LRELU_SLOPE)(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class Log(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Log, self).__init__(**kwargs)

    def call(self, x, x_mask, g=None, reverse=False, training=True):
        if not reverse:
            y = tf.log(tf.clip_by_value(x, 1e-5, tf.reduce_max(x))) * x_mask
            logdet = tf.reduce_sum(-y, axis=[2, 1])
            return y, logdet
        else:
            x = tf.exp(x) * x_mask
            return x


class Flip(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Flip, self).__init__(**kwargs)

    def call(self, x, mask, g=None, reverse=False, training=True):
        if not reverse:
            x = tf.reverse(x, [2])
            logdet = tf.zeros([tf.shape(x)[0]])
            return x, logdet
        else:
            x = tf.reverse(x, [2])
            return x


class ElementwiseAffine(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(ElementwiseAffine, self).__init__(**kwargs)
        self.channels = channels
        self.m = tf.Variable(tf.zeros([1, channels]))
        self.logs = tf.Variable(tf.zeros([1, channels]))

    def call(self, x, x_mask, g=None, reverse=False, training=True):
        if not reverse:
            y = self.m + tf.exp(self.logs) * x
            y = y * x_mask
            logdet = tf.reduce_sum(self.logs * x_mask, axis=[2, 1])
            return y, logdet
        else:
            x = (x - self.m) * tf.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(tf.keras.layers.Layer):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False,
                 **kwargs):
        super(ResidualCouplingLayer, self).__init__(**kwargs)
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = tf.keras.layers.Conv1D(hidden_channels, 1, padding='SAME')
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = tf.keras.layers.Conv1D(self.half_channels * (2 - mean_only), 1,
                                           kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, x, x_mask, g=None, reverse=False, training=True):
        x0, x1 = x[:, :, :self.half_channels], x[:, :, self.half_channels:]
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g, training=training)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = stats[:, :, :self.half_channels], x[:, :, self.half_channels:]
        else:
            m = stats
            logs = tf.zeros_like(m)

        if not reverse:
            x1 = m + x1 * tf.exp(logs) * x_mask
            x = tf.concat([x0, x1], 2)
            logdet = tf.reduce_sum(logs, axis=[2, 1])
            return x, logdet
        else:
            x1 = (x1 - m) * tf.exp(-logs) * x_mask
            x = tf.concat([x0, x1], 2)
            return x


class DDSConv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, n_layers, p_dropout=0., **kwargs):
        super(DDSConv, self).__init__(**kwargs)

        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = tf.keras.layers.Dropout(p_dropout)
        self.convs_sep = []
        self.convs_1x1 = []
        self.norms_1 = []
        self.norms_2 = []

        for i in range(n_layers):
            dilation = kernel_size ** i
            self.convs_sep.append(tf.keras.layers.DepthwiseConv2D((self.kernel_size, 1), padding='SAME',
                                                                  dilation_rate=dilation))
            self.convs_1x1.append(tf.keras.layers.Conv1D(channels, 1, padding='SAME'))
            self.norms_1.append(tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5))
            self.norms_2.append(tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5))

    def call(self, x, x_mask, g=None, training=True):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](tf.expand_dims(x * x_mask, 2))[:, :, 0]
            y = self.norms_1[i](y, training=training)
            y = gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y, training=training)
            y = gelu(y)
            y = self.drop(y, training=training)
            x = x + y
        return x * x_mask


class ConvFlow(tf.keras.layers.Layer):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0, **kwargs):
        super(ConvFlow, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = tf.keras.layers.Conv1D(filter_channels, 1, padding='SAME')
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
        self.proj = tf.keras.layers.Conv1D(self.half_channels * (num_bins * 3 - 1), 1,
                                           kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, x, x_mask, g=None, reverse=False, training=True):
        x0, x1 = x[:, :, :self.half_channels], x[:, :, self.half_channels:]
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g, training=training)
        h = self.proj(h) * x_mask

        x1 = tf.transpose(x1, [0, 2, 1])
        h = tf.transpose(h, [0, 2, 1])
        b, c, t = shape_list(x1)
        h = tf.transpose(tf.reshape(h, [b, c, -1, t]), [0, 1, 3, 2])

        unnormalized_widths = h[:, :, :, :self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[:, :, :, self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[:, :, :, 2 * self.num_bins:]

        x1, logabsdet = piecewise_rational_quadratic_transform_tf(x1,
                                                                  unnormalized_widths,
                                                                  unnormalized_heights,
                                                                  unnormalized_derivatives,
                                                                  inverse=reverse,
                                                                  tails='linear',
                                                                  tail_bound=self.tail_bound
                                                                  )
        x1 = tf.transpose(x1, [0, 2, 1])
        logabsdet = tf.transpose(logabsdet, [0, 2, 1])
        x = tf.concat([x0, x1], 2) * x_mask
        logdet = tf.reduce_sum(logabsdet * x_mask, axis=[2, 1])
        if not reverse:
            return x, logdet
        else:
            return x
