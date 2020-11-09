import tensorflow as tf
import math
from .layer import (
    glu,
    BLSTM,
    downsample,
    center_trim,
    Conv1DTranspose,
    ConvScaling,
)
from malaya_speech.utils.tf_featurization import pad_and_partition


class Model:
    def __init__(
        self,
        inputs,
        sources = 4,
        audio_channels = 1,
        channels = 64,
        depth = 6,
        rewrite = True,
        use_glu = True,
        rescale = 0.1,
        kernel_size = 8,
        stride = 4,
        growth = 2.0,
        lstm_layers = 2,
        context = 3,
        partition_length = 44100 * 2,
        norm_after_partition = False,
        output_shape_same_as_input = False,
        logging = False,
        kernel_initializer = ConvScaling,
    ):
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.partition_length = partition_length

        if use_glu:
            activation = glu
            ch_scale = 2
        else:
            activation = tf.nn.relu
            ch_scale = 1

        in_channels = audio_channels

        self.encoder, self.decoder = [], []
        for index in range(depth):
            encoder = tf.keras.Sequential()
            encoder.add(
                tf.keras.layers.Conv1D(
                    channels,
                    kernel_size,
                    stride,
                    activation = tf.nn.relu,
                    kernel_initializer = kernel_initializer,
                )
            )
            if rewrite:
                encoder.add(
                    tf.keras.layers.Conv1D(
                        ch_scale * channels,
                        1,
                        activation = activation,
                        kernel_initializer = kernel_initializer,
                    )
                )
            self.encoder.append(encoder)

            if index > 0:
                out_channels = in_channels
            else:
                out_channels = sources * audio_channels

            decoder = tf.keras.Sequential()
            if rewrite:
                decoder.add(
                    tf.keras.layers.Conv1D(
                        ch_scale * channels,
                        context,
                        activation = activation,
                        kernel_initializer = kernel_initializer,
                    )
                )

            if index > 0:
                a = tf.nn.relu
            else:
                a = None

            decoder.add(
                Conv1DTranspose(
                    out_channels,
                    kernel_size,
                    stride,
                    activation = a,
                    kernel_initializer = kernel_initializer,
                )
            )
            self.decoder.insert(0, decoder)
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        if partition_length:
            partitioned = pad_and_partition(inputs, self.partition_length)
            if norm_after_partition:
                mean = tf.reduce_mean(partitioned, axis = 0)
                std = tf.math.reduce_std(partitioned, axis = 0)
                partitioned = (partitioned - mean) / std

        valid_length = self.valid_length(partitioned.shape.as_list()[1])
        delta = valid_length - self.partition_length
        padded = tf.pad(
            partitioned,
            [[0, 0], [delta // 2, delta - delta // 2], [0, 0]],
            'CONSTANT',
        )

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        x = padded
        saved = [x]
        for encode in self.encoder:
            if logging:
                print(x)
            x = encode(x)
            saved.append(x)

        if logging:
            print('x', x)
        if self.lstm:
            x = self.lstm(x)

        for decode in self.decoder:
            if logging:
                print(x)
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        if logging:
            print('x', x)

        self.logits = x
        self.logits = tf.reshape(self.logits, (-1, self.sources))
        if output_shape_same_as_input:
            self.logits = self.logits[: tf.shape(inputs)[0]]

    def valid_length(self, length):
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        return int(length)
