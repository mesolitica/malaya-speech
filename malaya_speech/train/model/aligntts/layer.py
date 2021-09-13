from ..conformer.layer import PositionalEncoding, MultiHeadAttention
import tensorflow as tf


class FFTransformer(tf.keras.layers.Layer):
    def __init__(self, in_out_channels, num_heads,
                 hidden_channels_ffn=1024,
                 kernel_size_fft=3,
                 dropout_p=0.1,
                 **kwargs,):
        super(FFTransformer, self).__init__(**kwargs)
        self.self_attn = MultiHeadAttention(num_heads=num_heads,
                                            head_size=in_out_channels, dropout=dropout_p,
                                            return_attn_coef=True)
        self.conv1 = tf.keras.layers.Conv1D(
            filters=hidden_channels_ffn,
            kernel_size=kernel_size_fft,
            strides=1,
            padding='same',
            name='conv_1',
            activation=tf.nn.relu,
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=in_out_channels,
            kernel_size=kernel_size_fft,
            strides=1,
            padding='same',
            name='conv_2',
        )
        self.norm1 = tf.keras.layers.LayerNormalization(name='LayerNorm_1')
        self.norm2 = tf.keras.layers.LayerNormalization(name='LayerNorm_2')
        self.dropout1 = tf.keras.layers.Dropout(dropout_p)
        self.dropout2 = tf.keras.layers.Dropout(dropout_p)

    def call(self, src, training=True):
        src2, enc_align = self.self_attn([src, src, src], training=training)
        src = src + self.dropout1(src2, training=training)
        src = self.norm1(src + src2)
        src2 = self.conv2(self.conv1(src, training=training), training=training)
        src2 = self.dropout2(src2, training=training)
        src = src + src2
        src = self.norm2(src)
        return src, enc_align


class MDNBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, **kwargs,):
        super(MDNBlock, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv1D(
            filters=in_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            name='conv_1',
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=out_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            name='conv_2',
        )
        self.norm1 = tf.keras.layers.LayerNormalization(name='LayerNorm_1')
        self.dropout1 = tf.keras.layers.Dropout(0.1)
