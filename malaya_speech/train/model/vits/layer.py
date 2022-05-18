import tensorflow as tf
from ..fastspeech.layer import gelu


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
            self.norms_1.append(tf.keras.layers.LayerNormalization(axis=-1))
            self.norms_2.append(tf.keras.layers.LayerNormalization(axis=-1))

    def call(self, x, x_mask, training=True):
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y, training=training)
            y = gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y, training=training)
            y = gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask
