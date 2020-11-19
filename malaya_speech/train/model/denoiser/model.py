import tensorflow as tf
import math
from ..demucs.layer import (
    glu,
    downsample,
    center_trim,
    Conv1DTranspose,
    ConvScaling,
)
from .layer import downsample2, upsample2, BLSTM
from malaya_speech.utils.tf_featurization import pad_and_partition


class Model:
    def __init__(
        self,
        inputs,
        y = None,
        chin = 1,
        chout = 1,
        hidden = 48,
        depth = 5,
        use_glu = True,
        kernel_size = 8,
        stride = 4,
        causal = True,
        resample = 4,
        growth = 2,
        max_hidden = 10000,
        normalize = True,
        rescale = 0.1,
        floor = 1e-3,
        lstm_layers = 2,
        partition_length = 44100 * 2,
        norm_after_partition = False,
        logging = False,
        kernel_initializer = ConvScaling,
    ):
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.partition_length = partition_length

        if use_glu:
            activation = glu
            ch_scale = 2
        else:
            activation = tf.nn.relu
            ch_scale = 1

        self.encoder, self.decoder = [], []
        for index in range(depth):
            encoder = tf.keras.Sequential()
            encoder.add(
                tf.keras.layers.Conv1D(
                    hidden,
                    kernel_size,
                    stride,
                    activation = tf.nn.relu,
                    kernel_initializer = kernel_initializer,
                )
            )
            encoder.add(
                tf.keras.layers.Conv1D(
                    ch_scale * hidden,
                    1,
                    activation = activation,
                    kernel_initializer = kernel_initializer,
                )
            )
            self.encoder.append(encoder)

            decoder = tf.keras.Sequential()
            decoder.add(
                tf.keras.layers.Conv1D(
                    ch_scale * hidden,
                    1,
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
                    chout,
                    kernel_size,
                    stride,
                    activation = a,
                    kernel_initializer = kernel_initializer,
                )
            )
            self.decoder.insert(0, decoder)
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi = not causal)

        if self.normalize:
            mono = tf.reduce_mean(inputs, axis = 1, keepdims = True)
            self.std = tf.math.reduce_std(mono, axis = 0, keepdims = True)
            inputs = inputs / (self.floor + self.std)
        else:
            self.std = 1.0

        partitioned = pad_and_partition(inputs, self.partition_length)
        if norm_after_partition:
            mean = tf.reduce_mean(partitioned, axis = 0)
            std = tf.math.reduce_std(partitioned, axis = 0)
            partitioned = (partitioned - mean) / std

        valid_length = self.valid_length(self.partition_length)
        delta = valid_length - self.partition_length
        padded = tf.pad(partitioned, [[0, 0], [0, delta], [0, 0]], 'CONSTANT')
        x = padded
        if logging:
            print(x)
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        if logging:
            print(x)
        skips = []
        for encode in self.encoder:
            if logging:
                print(x)
            x = encode(x)
            skips.append(x)
        if logging:
            print('x', x)
        x = self.lstm(x)
        for decode in self.decoder:
            if logging:
                print(x)
            skip = skips.pop(-1)
            x = x + skip[:, : tf.shape(x)[1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        if logging:
            print('x', x)

        self.logits = x
        self.logits = tf.reshape(self.logits, (-1, self.chout))
        if y is not None:
            self.logits = self.logits[: tf.shape(y)[0]]
        else:
            self.logits = self.logits[: tf.shape(inputs)[0]]
        self.logits = self.std * self.logits

    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)
