import tensorflow as tf
from .layer import gelu


def bilstm(dim, layers = 1):
    a


class Model:
    def __init__(
        self,
        inputs,
        sources = 4,
        audio_channels = 1,
        channels = 64,
        depth = 6,
        rewrite = True,
        use_gelu = True,
        upsample = False,
        rescale = 0.1,
        kernel_size = 8,
        stride = 4,
        growth = 2.0,
        lstm_layers = 2,
        context = 3,
    ):

        if use_gelu:
            activation = gelu
            ch_scale = 2
        else:
            activation = tf.nn.relu
            ch_scale = 1

        in_channels = audio_channels
        for index in range(depth):
            encode = []
