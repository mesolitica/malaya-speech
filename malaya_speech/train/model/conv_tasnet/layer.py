# https://github.com/kaparoo/Conv-TasNet

import tensorflow as tf
from .norm import GlobalLayerNorm as gLN
from .norm import CausalLayerNorm as cLN


class Encoder(tf.keras.layers.Layer):

    def __init__(self, param, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)

        self.U = tf.keras.layers.Dense(units=param.N, activation='relu')

    def call(self, mixture_segments):
        # (, That, L) -> (, That, N)
        return self.U(mixture_segments)


class Decoder(tf.keras.layers.Layer):

    def __init__(self, param, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)

        self.B = tf.keras.layers.Dense(units=param.L)

    def call(self, source_weights):
        # (, C, That, N) -> (, C, That, L)
        return self.B(source_weights)  # estimated_sources


class Separator(tf.keras.layers.Layer):

    def __init__(self, param, **kwargs):
        super(Separater, self).__init__(name='Separation', **kwargs)

        self.normalization = tf.keras.layers.LayerNormalization()

        self.conv1x1_in = tf.keras.layers.Dense(units=param.B)

        # Dilated-TCN
        self.conv1d_blocks = []
        for r in range(param.R):
            for x in range(param.X):
                self.conv1d_blocks.append(Conv1DBlock(param, r, x))
        self.conv1d_blocks[-1].is_last = True
        self.skip_connection = tf.keras.layers.Add()

        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.conv1x1_out = tf.keras.layers.Dense(units=param.C*param.N,
                                                 activation='sigmoid')

        self.reshape_mask = tf.keras.layers.Reshape(
            target_shape=[param.That, param.C, param.N])

        self.reorder_mask = tf.keras.layers.Permute([2, 1, 3])

    def call(self, mixture_weights):
        # (, That, N) -> (, That, N)
        normalized_weights = self.normalization(mixture_weights)

        # (, That, N) -> (, That, B)
        block_inputs = self.conv1x1_in(normalized_weights)

        # (, That, B) -> (, That, Sc)
        skip_outputs = []
        for conv1d_block in self.conv1d_blocks:
            _skip_outputs, _block_outputs = conv1d_block(block_inputs)
            block_inputs = _block_outputs
            skip_outputs.append(_skip_outputs)
        tcn_outputs = self.skip_connection(skip_outputs)
        tcn_outputs = self.prelu(tcn_outputs)

        # (, That, Sc) -> (, That, C*N)
        source_masks = self.conv1x1_out(tcn_outputs)

        # (, That, C*N) -> (, C, That, N)
        source_masks = self.reorder_mask(self.reshape_mask(source_masks))
        return source_masks


class Conv1DBlock(tf.keras.layers.Layer):

    def __init__(self, param, r: int, x: int, **kwargs):
        super(Conv1DBlock, self).__init__(
            name=f'conv1d_block_r{r}_x{x}', **kwargs)

        self.is_last = False
        self.B = param.B

        self.conv1x1_bottle = tf.keras.layers.Dense(units=param.H)
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])

        padding_label: str = None
        if param.causal:
            padding_label = 'causal'
            self.normalization1 = cLN()
            self.normalization2 = cLN()
        else:
            padding_label = 'same'
            self.normalization1 = gLN()
            self.normalization2 = gLN()

        self.dconv = tf.keras.layers.Conv1D(filters=param.H,
                                            kernel_size=param.P,
                                            dilation_rate=2**x,
                                            padding=padding_label,
                                            groups=param.H)

        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.conv1x1_skipconn = tf.keras.layers.Dense(units=param.Sc)

    def build(self, input_shape):
        if not self.is_last:
            self.conv1x1_residual = tf.keras.layers.Dense(units=self.B)
            self.link_residual = tf.keras.layers.Add()

    def call(self, block_inputs):
        # (, That, B) -> (, That, H)
        block_outputs = self.conv1x1_bottle(block_inputs)

        # (, That, H) -> (, That, H)
        block_outputs = self.prelu1(block_outputs)
        # (, That, H) -> (, That, H)
        block_outputs = self.normalization1(block_outputs)

        # (, That, H) -> (, That, H)
        block_outputs = self.dconv(block_outputs)

        # (, That, H) -> (, That, H)
        block_outputs = self.prelu2(block_outputs)
        # (, That, H) -> (, That, H)
        block_outputs = self.normalization2(block_outputs)

        # (, That, H) -> (, That, Sc)
        skipconn_outputs = self.conv1x1_skipconn(block_outputs)

        residual_outputs = block_inputs
        if not self.is_last:
            # (, That, H) -> (, That, B)
            block_outputs = self.conv1x1_residual(block_outputs)
            residual_outputs = self.link_residual(
                [block_inputs, block_outputs])

        return skipconn_outputs, residual_outputs
