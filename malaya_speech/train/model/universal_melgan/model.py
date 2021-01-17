import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import (
    activations,
    constraints,
    initializers,
    regularizers,
)
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import Conv1D, SeparableConv1D
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn, nn_ops
from ..melgan.layer import WeightNormalization, GroupConv1D
from ..melgan.model import Discriminator as WaveFormDiscriminator


def get_initializer(initializer_seed = 42):
    """Creates a `tf.initializers.glorot_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        GlorotNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.glorot_normal(seed = initializer_seed)


class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type = 'REFLECT', **kwargs):
        """Initialize TFReflectionPad1d module.
        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(
            x,
            [[0, 0], [self.padding_size, self.padding_size], [0, 0]],
            self.padding_type,
        )


class TFReflectionPad2d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type = 'REFLECT', **kwargs):
        """Initialize TFReflectionPad2d module.
        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, W, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, W, C).
        """
        return tf.pad(
            x,
            [[0, 0], [self.padding_size, self.padding_size], [0, 0], [0, 0]],
            self.padding_type,
        )


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters = filters,
            kernel_size = (kernel_size, 1),
            strides = (strides, 1),
            padding = 'same',
            kernel_initializer = get_initializer(initializer_seed),
        )
        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack module."""

    def __init__(
        self,
        kernel_size,
        filters,
        dilation_rate,
        use_bias,
        nonlinear_activation,
        nonlinear_activation_params,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
        """Initialize TFResidualStack module.
        Args:
            kernel_size (int): Kernel size.
            filters (int): Number of filters.
            dilation_rate (int): Dilation rate.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__(**kwargs)
        self.blocks = [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.Conv1D(
                filters = filters,
                kernel_size = kernel_size,
                dilation_rate = dilation_rate,
                use_bias = use_bias,
                kernel_initializer = get_initializer(initializer_seed),
            ),
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            tf.keras.layers.Conv1D(
                filters = filters,
                kernel_size = 1,
                use_bias = use_bias,
                kernel_initializer = get_initializer(initializer_seed),
            ),
        ]
        self.shortcut = tf.keras.layers.Conv1D(
            filters = filters * 2,
            kernel_size = 1,
            use_bias = use_bias,
            kernel_initializer = get_initializer(initializer_seed),
            name = 'shortcut',
        )

        # apply weightnorm
        if is_weight_norm:
            self._apply_weightnorm(self.blocks)
            self.shortcut = WeightNormalization(self.shortcut)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        _x = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        shortcut = self.shortcut(x)
        # GAU
        l, r = tf.split(shortcut, 2, axis = -1)
        l = tf.nn.sigmoid(l)
        r = tf.nn.tanh(r)
        shortcut = l * r
        return shortcut + _x

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if 'conv1d' in layer_name or 'dense' in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class Generator(tf.keras.Model):
    """Tensorflow MelGAN generator module."""

    def __init__(self, config, **kwargs):
        """Initialize TFMelGANGenerator module.
        Args:
            config: config object of Melgan generator.
        """
        super().__init__(**kwargs)

        # check hyper parameter is valid or not
        assert config.filters >= np.prod(config.upsample_scales)
        assert config.filters % (2 ** len(config.upsample_scales)) == 0

        # add initial layer
        layers = []
        layers += [
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type = config.padding_type,
                name = 'first_reflect_padding',
            ),
            tf.keras.layers.Conv1D(
                filters = config.filters,
                kernel_size = config.kernel_size,
                use_bias = config.use_bias,
                kernel_initializer = get_initializer(config.initializer_seed),
            ),
        ]

        for i, upsample_scale in enumerate(config.upsample_scales):
            # add upsampling layer
            layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(
                    **config.nonlinear_activation_params
                ),
                TFConvTranspose1d(
                    filters = config.filters // (2 ** (i + 1)),
                    kernel_size = upsample_scale * 2,
                    strides = upsample_scale,
                    padding = 'same',
                    is_weight_norm = config.is_weight_norm,
                    initializer_seed = config.initializer_seed,
                    name = 'conv_transpose_._{}'.format(i),
                ),
            ]

            # ad residual stack layer
            for j in range(config.stacks):
                layers += [
                    TFResidualStack(
                        kernel_size = config.stack_kernel_size,
                        filters = config.filters // (2 ** (i + 1)),
                        dilation_rate = config.stack_kernel_size ** j,
                        use_bias = config.use_bias,
                        nonlinear_activation = config.nonlinear_activation,
                        nonlinear_activation_params = config.nonlinear_activation_params,
                        is_weight_norm = config.is_weight_norm,
                        initializer_seed = config.initializer_seed,
                        name = 'residual_stack_._{}._._{}'.format(i, j),
                    )
                ]
        # add final layer
        layers += [
            getattr(tf.keras.layers, config.nonlinear_activation)(
                **config.nonlinear_activation_params
            ),
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type = config.padding_type,
                name = 'last_reflect_padding',
            ),
            tf.keras.layers.Conv1D(
                filters = config.out_channels,
                kernel_size = config.kernel_size,
                use_bias = config.use_bias,
                kernel_initializer = get_initializer(config.initializer_seed),
            ),
        ]
        if config.use_final_nolinear_activation:
            layers += [tf.keras.layers.Activation('tanh')]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.melgan = tf.keras.models.Sequential(layers)

    def call(self, mels, **kwargs):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, T, channels)
        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        return self.melgan(mels)

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if 'conv1d' in layer_name or 'dense' in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass

    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape = [1, 100, 80], dtype = tf.float32)
        self(fake_mels)


class STFTDiscriminator(tf.keras.layers.Layer):
    """Tensorflow MelGAN generator module."""

    def __init__(
        self,
        fft_length = 1024,
        frame_length = 1024,
        frame_step = 256,
        out_channels = 1,
        kernel_sizes = [5, 3],
        filters = 32,
        max_downsample_filters = 32,
        use_bias = True,
        downsample_scales = [1, 1, 1],
        nonlinear_activation = 'LeakyReLU',
        nonlinear_activation_params = {'alpha': 0.2},
        padding_type = 'REFLECT',
        is_weight_norm = True,
        initializer_seed = 0.02,
        **kwargs
    ):
        """Initilize MelGAN discriminator module.
        Args:
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15.
                the last two layers' kernel size will be 5 and 3, respectively.
            filters (int): Initial number of filters for conv layer.
            max_downsample_filters (int): Maximum number of filters for downsampling layers.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            padding_type (str): Padding type (support only "REFLECT", "CONSTANT", "SYMMETRIC")
        """
        super().__init__(**kwargs)
        discriminator = []

        self.fft_length = fft_length
        self.frame_length = frame_length
        self.frame_step = frame_step

        # check kernel_size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        discriminator = [
            TFReflectionPad2d(
                (np.prod(kernel_sizes) - 1) // 2, padding_type = padding_type
            ),
            tf.keras.layers.Conv2D(
                filters = filters,
                kernel_size = int(np.prod(kernel_sizes)),
                use_bias = use_bias,
                kernel_initializer = get_initializer(initializer_seed),
            ),
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
        ]

        # add downsample layers
        in_chs = filters
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_filters)
            discriminator += [
                tf.keras.layers.Conv2D(
                    filters = out_chs,
                    kernel_size = downsample_scale * 10 + 1,
                    strides = [downsample_scale, 2],
                    padding = 'same',
                    use_bias = use_bias,
                    kernel_initializer = get_initializer(initializer_seed),
                )
            ]
            discriminator += [
                getattr(tf.keras.layers, nonlinear_activation)(
                    **nonlinear_activation_params
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_filters)
        discriminator += [
            tf.keras.layers.Conv2D(
                filters = out_chs,
                kernel_size = kernel_sizes[0],
                padding = 'same',
                use_bias = use_bias,
                kernel_initializer = get_initializer(initializer_seed),
            )
        ]
        discriminator += [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            )
        ]
        discriminator += [
            tf.keras.layers.Conv2D(
                filters = out_channels,
                kernel_size = kernel_sizes[1],
                padding = 'same',
                use_bias = use_bias,
                kernel_initializer = get_initializer(initializer_seed),
            )
        ]

        if is_weight_norm is True:
            self._apply_weightnorm(discriminator)

        self.disciminator = discriminator

    def call(self, x, **kwargs):
        outs = []
        x = tf.abs(
            tf.signal.stft(
                signals = x,
                frame_length = self.frame_length,
                frame_step = self.frame_step,
                fft_length = self.fft_length,
            )
        )
        x = tf.expand_dims(x, -1)
        for f in self.disciminator:
            x = f(x)
            outs += [x]
        return outs

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if 'conv1d' in layer_name or 'dense' in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class MultiScaleDiscriminator(tf.keras.Model):
    """MelGAN multi-scale discriminator module."""

    def __init__(self, waveform_config, stft_config, **kwargs):
        super().__init__(**kwargs)
        self.waveform_discriminator = []
        self.stft_discriminator = []

        # add discriminator
        for i in range(waveform_config.scales):
            self.waveform_discriminator += [
                WaveFormDiscriminator(
                    out_channels = waveform_config.out_channels,
                    kernel_sizes = waveform_config.kernel_sizes,
                    filters = waveform_config.filters,
                    max_downsample_filters = waveform_config.max_downsample_filters,
                    use_bias = waveform_config.use_bias,
                    downsample_scales = waveform_config.downsample_scales,
                    nonlinear_activation = waveform_config.nonlinear_activation,
                    nonlinear_activation_params = waveform_config.nonlinear_activation_params,
                    padding_type = waveform_config.padding_type,
                    is_weight_norm = waveform_config.is_weight_norm,
                    initializer_seed = waveform_config.initializer_seed,
                    name = 'waveform_discriminator_scale_._{}'.format(i),
                )
            ]
            self.waveform_pooling = getattr(
                tf.keras.layers, waveform_config.downsample_pooling
            )(**waveform_config.downsample_pooling_params)

        for i in range(stft_config.scales):
            self.stft_discriminator += [
                STFTDiscriminator(
                    fft_length = stft_config.fft_length[i],
                    frame_length = stft_config.frame_length[i],
                    frame_step = stft_config.frame_step[i],
                    out_channels = stft_config.out_channels,
                    kernel_sizes = stft_config.kernel_sizes,
                    filters = stft_config.filters,
                    max_downsample_filters = stft_config.max_downsample_filters,
                    use_bias = stft_config.use_bias,
                    downsample_scales = stft_config.downsample_scales,
                    nonlinear_activation = stft_config.nonlinear_activation,
                    nonlinear_activation_params = stft_config.nonlinear_activation_params,
                    padding_type = stft_config.padding_type,
                    is_weight_norm = stft_config.is_weight_norm,
                    initializer_seed = stft_config.initializer_seed,
                    name = 'stft_discriminator_scale_._{}'.format(i),
                )
            ]

    def call(self, x, **kwargs):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.stft_discriminator:
            outs += [f(x[:, :, 0])]

        for f in self.waveform_discriminator:
            outs += [f(x)]
            x = self.waveform_pooling(x)

        return outs
