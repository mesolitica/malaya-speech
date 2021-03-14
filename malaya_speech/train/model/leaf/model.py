# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np
import math
from . import normalization


class PreempInit(tf.keras.initializers.Initializer):
    """Keras initializer for the pre-emphasis.
  Returns a Tensor to initialize the pre-emphasis layer of a Leaf instance.
  Attributes:
    alpha: parameter that controls how much high frequencies are emphasized by
      the following formula output[n] = input[n] - alpha*input[n-1] with 0 <
      alpha < 1 (higher alpha boosts high frequencies)
  """

    def __init__(self, alpha = 0.97):
        self.alpha = alpha

    def __call__(self, shape, dtype = None):
        assert shape == (
            2,
            1,
            1,
        ), 'Cannot initialize preemp layer of size {}'.format(shape)
        preemp_arr = np.zeros(shape)
        preemp_arr[0, 0, 0] = -self.alpha
        preemp_arr[1, 0, 0] = 1
        return tf.convert_to_tensor(preemp_arr, dtype = dtype)

    def get_config(self):
        return self.__dict__


class GaborInit(tf.keras.initializers.Initializer):
    """Keras initializer for the complex-valued convolution.
  Returns a Tensor to initialize the complex-valued convolution layer of a
  Leaf instance with Gabor filters designed to match the
  frequency response of standard mel-filterbanks.
  If the shape has rank 2, this is a complex convolution with filters only
  parametrized by center frequency and FWHM, so we initialize accordingly.
  In this case, we define the window len as 401 (default value), as it is not
  used for initialization.
  """

    def __init__(self, **kwargs):
        kwargs.pop('n_filters', None)
        self._kwargs = kwargs

    def __call__(self, shape, dtype = None):
        n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
        window_len = 401 if len(shape) == 2 else shape[0]
        gabor_filters = Gabor(
            n_filters = n_filters, window_len = window_len, **self._kwargs
        )
        if len(shape) == 2:
            return gabor_filters.gabor_params_from_mels
        else:
            even_indices = tf.range(shape[2], delta = 2)
            odd_indices = tf.range(start = 1, limit = shape[2], delta = 2)
            filters = gabor_filters.gabor_filters
            filters_real_and_imag = tf.dynamic_stitch(
                [even_indices, odd_indices],
                [tf.math.real(filters), tf.math.imag(filters)],
            )
            return tf.transpose(
                filters_real_and_imag[:, tf.newaxis, :], [2, 1, 0]
            )

    def get_config(self):
        return self._kwargs


class ExponentialMovingAverage(tf.keras.layers.Layer):
    """Computes of an exponential moving average of an sequential input."""

    def __init__(
        self, coeff_init, per_channel: bool = False, trainable: bool = False
    ):
        """Initializes the ExponentialMovingAverage.
    Args:
      coeff_init: the value of the initial coeff.
      per_channel: whether the smoothing should be different per channel.
      trainable: whether the smoothing should be trained or not.
    """
        super().__init__(name = 'EMA')
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self._trainable = trainable

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self._weights = self.add_weight(
            name = 'smooth',
            shape = (num_channels,) if self._per_channel else (1,),
            initializer = tf.keras.initializers.Constant(self._coeff_init),
            trainable = self._trainable,
        )

    def call(self, inputs: tf.Tensor, initial_state: tf.Tensor):
        """Inputs is of shape [batch, seq_length, num_filters]."""
        w = tf.clip_by_value(
            self._weights, clip_value_min = 0.0, clip_value_max = 1.0
        )
        result = tf.scan(
            lambda a, x: w * x + (1.0 - w) * a,
            tf.transpose(inputs, (1, 0, 2)),
            initializer = initial_state,
        )
        return tf.transpose(result, (1, 0, 2))


class PCENLayer(tf.keras.layers.Layer):
    """Per-Channel Energy Normalization.
  This applies a fixed or learnable normalization by an exponential moving
  average smoother, and a compression.
  See https://arxiv.org/abs/1607.05666 for more details.
  """

    def __init__(
        self,
        alpha: float = 0.96,
        smooth_coef: float = 0.04,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-6,
        trainable: bool = False,
        learn_smooth_coef: bool = False,
        per_channel_smooth_coef: bool = False,
        name = 'PCEN',
    ):
        """PCEN constructor.
    Args:
      alpha: float, exponent of EMA smoother
      smooth_coef: float, smoothing coefficient of EMA
      delta: float, bias added before compression
      root: float, one over exponent applied for compression (r in the paper)
      floor: float, offset added to EMA smoother
      trainable: bool, False means fixed_pcen, True is trainable_pcen
      learn_smooth_coef: bool, True means we also learn the smoothing
        coefficient
      per_channel_smooth_coef: bool, True means each channel has its own smooth
        coefficient
      name: str, name of the layer
    """
        super().__init__(name = name)
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable
        self._learn_smooth_coef = learn_smooth_coef
        self._per_channel_smooth_coef = per_channel_smooth_coef

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.alpha = self.add_weight(
            name = 'alpha',
            shape = [num_channels],
            initializer = tf.keras.initializers.Constant(self._alpha_init),
            trainable = self._trainable,
        )
        self.delta = self.add_weight(
            name = 'delta',
            shape = [num_channels],
            initializer = tf.keras.initializers.Constant(self._delta_init),
            trainable = self._trainable,
        )
        self.root = self.add_weight(
            name = 'root',
            shape = [num_channels],
            initializer = tf.keras.initializers.Constant(self._root_init),
            trainable = self._trainable,
        )
        if self._learn_smooth_coef:
            self.ema = ExponentialMovingAverage(
                coeff_init = self._smooth_coef,
                per_channel = self._per_channel_smooth_coef,
                trainable = True,
            )
        else:
            self.ema = tf.keras.layers.SimpleRNN(
                units = num_channels,
                activation = None,
                use_bias = False,
                kernel_initializer = tf.keras.initializers.Identity(
                    gain = self._smooth_coef
                ),
                recurrent_initializer = tf.keras.initializers.Identity(
                    gain = 1.0 - self._smooth_coef
                ),
                return_sequences = True,
                trainable = False,
            )

    def call(self, inputs):
        alpha = tf.math.minimum(self.alpha, 1.0)
        root = tf.math.maximum(self.root, 1.0)
        ema_smoother = self.ema(
            inputs, initial_state = tf.gather(inputs, 0, axis = 1)
        )
        one_over_root = 1.0 / root
        output = (
            inputs / (self._floor + ema_smoother) ** alpha + self.delta
        ) ** one_over_root - self.delta ** one_over_root
        return output


def gabor_impulse_response(
    t: tf.Tensor, center: tf.Tensor, fwhm: tf.Tensor
) -> tf.Tensor:
    """Computes the gabor impulse response."""
    denominator = 1.0 / (tf.math.sqrt(2.0 * math.pi) * fwhm)
    gaussian = tf.exp(tf.tensordot(1.0 / (2.0 * fwhm ** 2), -t ** 2, axes = 0))
    center_frequency_complex = tf.cast(center, tf.complex64)
    t_complex = tf.cast(t, tf.complex64)
    sinusoid = tf.math.exp(
        1j * tf.tensordot(center_frequency_complex, t_complex, axes = 0)
    )
    denominator = tf.cast(denominator, dtype = tf.complex64)[:, tf.newaxis]
    gaussian = tf.cast(gaussian, dtype = tf.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters_function(kernel, size: int = 401) -> tf.Tensor:
    """Computes the gabor filters from its parameters for a given size.
  Args:
    kernel: tf.Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.
  Returns:
    A tf.Tensor<float>[filters, size].
  """
    return gabor_impulse_response(
        tf.range(-(size // 2), (size + 1) // 2, dtype = tf.float32),
        center = kernel[:, 0],
        fwhm = kernel[:, 1],
    )


class SquaredModulus(tf.keras.layers.Layer):
    """Squared modulus layer.
  Returns a keras layer that implements a squared modulus operator.
  To implement the squared modulus of C complex-valued channels, the expected
  input dimension is N*1*W*(2*C) where channels role alternates between
  real and imaginary part.
  The way the squared modulus is computed is real ** 2 + imag ** 2 as follows:
  - squared operator on real and imag
  - average pooling to compute (real ** 2 + imag ** 2) / 2
  - multiply by 2
  Attributes:
    pool: average-pooling function over the channel dimensions
  """

    def __init__(self):
        super().__init__(name = 'squared_modulus')
        # self._pool = tf.keras.layers.AveragePooling1D(
        #     pool_size = 2, strides = 2
        # )

    def call(self, x):
        x = tf.transpose(x, perm = [0, 2, 1])
        output = 2 * tf.nn.avg_pool1d(x ** 2, 2, 2, padding = 'VALID')
        # output = 2 * self._pool(x ** 2)
        return tf.transpose(output, perm = [0, 2, 1])


class GaborConstraint(tf.keras.constraints.Constraint):
    """Constraint mu and sigma, in radians.
  Mu is constrained in [0,pi], sigma s.t full-width at half-maximum of the
  gaussian response is in [1,pi/2]. The full-width at half maximum of the
  Gaussian response is 2*sqrt(2*log(2))/sigma . See Section 2.2 of
  https://arxiv.org/pdf/1711.01161.pdf for more details.
  """

    def __init__(self, kernel_size):
        """Initialize kernel size.
    Args:
      kernel_size: the length of the filter, in samples.
    """
        self._kernel_size = kernel_size

    def __call__(self, kernel):
        mu_lower = 0.0
        mu_upper = math.pi
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi
        clipped_mu = tf.clip_by_value(kernel[:, 0], mu_lower, mu_upper)
        clipped_sigma = tf.clip_by_value(kernel[:, 1], sigma_lower, sigma_upper)
        return tf.stack([clipped_mu, clipped_sigma], axis = 1)


class GaborConv1D(tf.keras.layers.Layer):
    """Implements a convolution with filters defined as complex Gabor wavelets.
  These filters are parametrized only by their center frequency and
  the full-width at half maximum of their frequency response.
  Thus, for n filters, there are 2*n parameters to learn.
  """

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
        input_shape,
        kernel_initializer,
        kernel_regularizer,
        name,
        trainable,
        sort_filters = False,
    ):
        super().__init__(name = name)
        self._filters = filters // 2
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._sort_filters = sort_filters
        # Weights are the concatenation of center freqs and inverse bandwidths.
        self._kernel = self.add_weight(
            name = 'kernel',
            shape = (self._filters, 2),
            initializer = kernel_initializer,
            regularizer = kernel_regularizer,
            trainable = trainable,
            constraint = GaborConstraint(self._kernel_size),
        )
        if self._use_bias:
            self._bias = self.add_weight(
                name = 'bias', shape = (self._filters * 2,)
            )

    def call(self, inputs):
        kernel = self._kernel.constraint(self._kernel)
        if self._sort_filters:
            filter_order = tf.argsort(kernel[:, 0])
            kernel = tf.gather(kernel, filter_order, axis = 0)
        filters = gabor_filters_function(kernel, self._kernel_size)
        real_filters = tf.math.real(filters)
        img_filters = tf.math.imag(filters)
        stacked_filters = tf.stack([real_filters, img_filters], axis = 1)
        stacked_filters = tf.reshape(
            stacked_filters, [2 * self._filters, self._kernel_size]
        )
        stacked_filters = tf.expand_dims(
            tf.transpose(stacked_filters, perm = (1, 0)), axis = 1
        )
        outputs = tf.nn.conv1d(
            inputs,
            stacked_filters,
            stride = self._strides,
            padding = self._padding,
        )
        if self._use_bias:
            outputs = tf.nn.bias_add(outputs, self._bias, data_format = 'NWC')
        return outputs


def gaussian_lowpass(sigma: tf.Tensor, filter_size: int):
    """Generates gaussian windows centered in zero, of std sigma.
  Args:
    sigma: tf.Tensor<float>[1, 1, C, 1] for C filters.
    filter_size: length of the filter.
  Returns:
    A tf.Tensor<float>[1, filter_size, C, 1].
  """
    sigma = tf.clip_by_value(
        sigma, clip_value_min = (2.0 / filter_size), clip_value_max = 0.5
    )
    t = tf.range(0, filter_size, dtype = tf.float32)
    t = tf.reshape(t, (1, filter_size, 1, 1))
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return tf.math.exp(-0.5 * (numerator / denominator) ** 2)


class GaussianLowpass(tf.keras.layers.Layer):
    """Depthwise pooling (each input filter has its own pooling filter).
  Pooling filters are parametrized as zero-mean Gaussians, with learnable
  std. They can be initialized with tf.keras.initializers.Constant(0.4)
  to approximate a Hanning window.
  We rely on depthwise_conv2d as there is no depthwise_conv1d in Keras so far.
  """

    def __init__(
        self,
        kernel_size,
        strides = 1,
        padding = 'same',
        use_bias = True,
        kernel_initializer = 'glorot_uniform',
        kernel_regularizer = None,
        trainable = False,
    ):

        super().__init__(name = 'learnable_pooling')
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.trainable = trainable

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name = 'kernel',
            shape = (1, 1, input_shape[2], 1),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            trainable = self.trainable,
        )

    def call(self, inputs):
        kernel = gaussian_lowpass(self.kernel, self.kernel_size)
        outputs = tf.expand_dims(inputs, axis = 1)
        outputs = tf.nn.depthwise_conv2d(
            outputs,
            kernel,
            strides = (1, self.strides, self.strides, 1),
            padding = self.padding.upper(),
        )
        return tf.squeeze(outputs, axis = 1)


class Gabor:
    """This class creates gabor filters designed to match mel-filterbanks.
  Attributes:
    n_filters: number of filters
    min_freq: minimum frequency spanned by the filters
    max_freq: maximum frequency spanned by the filters
    sample_rate: samplerate (samples/s)
    window_len: window length in samples
    n_fft: number of frequency bins to compute mel-filters
    normalize_energy: boolean, True means that all filters have the same energy,
      False means that the higher the center frequency of a filter, the higher
      its energy
  """

    def __init__(
        self,
        n_filters: int = 40,
        min_freq: float = 0.0,
        max_freq: float = 8000.0,
        sample_rate: int = 16000,
        window_len: int = 401,
        n_fft: int = 512,
        normalize_energy: bool = False,
    ):

        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy

    @property
    def gabor_params_from_mels(self):
        """Retrieves center frequencies and standard deviations of gabor filters."""
        coeff = tf.math.sqrt(2.0 * tf.math.log(2.0)) * self.n_fft
        sqrt_filters = tf.math.sqrt(self.mel_filters)
        center_frequencies = tf.cast(
            tf.argmax(sqrt_filters, axis = 1), dtype = tf.float32
        )
        peaks = tf.reduce_max(sqrt_filters, axis = 1, keepdims = True)
        half_magnitudes = peaks / 2.0
        fwhms = tf.reduce_sum(
            tf.cast(sqrt_filters >= half_magnitudes, dtype = tf.float32),
            axis = 1,
        )
        return tf.stack(
            [
                center_frequencies * 2 * np.pi / self.n_fft,
                coeff / (np.pi * fwhms),
            ],
            axis = 1,
        )

    def _mel_filters_areas(self, filters):
        """Area under each mel-filter."""
        peaks = tf.reduce_max(filters, axis = 1, keepdims = True)
        return (
            peaks
            * (
                tf.reduce_sum(
                    tf.cast(filters > 0, dtype = tf.float32),
                    axis = 1,
                    keepdims = True,
                )
                + 2
            )
            * np.pi
            / self.n_fft
        )

    @property
    def mel_filters(self):
        """Creates a bank of mel-filters."""
        # build mel filter matrix
        mel_filters = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = self.n_filters,
            num_spectrogram_bins = self.n_fft // 2 + 1,
            sample_rate = self.sample_rate,
            lower_edge_hertz = self.min_freq,
            upper_edge_hertz = self.max_freq,
        )
        mel_filters = tf.transpose(mel_filters, [1, 0])
        if self.normalize_energy:
            mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
        return mel_filters

    @property
    def gabor_filters(self):
        """Generates gabor filters that match the corresponding mel-filters."""
        gabor_filters = gabor_filters_function(
            self.gabor_params_from_mels, size = self.window_len
        )
        return gabor_filters * tf.cast(
            tf.math.sqrt(
                self._mel_filters_areas(self.mel_filters)
                * 2
                * tf.math.sqrt(np.pi)
                * self.gabor_params_from_mels[:, 1:2]
            ),
            dtype = tf.complex64,
        )


class Model(tf.keras.models.Model):
    """Keras layer that implements time-domain filterbanks.
  Creates a LEAF frontend, a learnable front-end that takes an audio
  waveform as input and outputs a learnable spectral representation. This layer
  can be initialized to replicate the computation of standard mel-filterbanks.
  A detailed technical description is presented in Section 3 of
  https://arxiv.org/abs/2101.08596 .
  """

    def __init__(
        self,
        learn_pooling: bool = True,
        learn_filters: bool = True,
        conv1d_cls = GaborConv1D,
        activation = SquaredModulus(),
        pooling_cls = GaussianLowpass,
        n_filters: int = 80,
        sample_rate: int = 16000,
        window_len: float = 25.0,
        window_stride: float = 10.0,
        compression_fn = PCENLayer(
            alpha = 0.96,
            smooth_coef = 0.04,
            delta = 2.0,
            floor = 1e-12,
            trainable = True,
            learn_smooth_coef = True,
            per_channel_smooth_coef = True,
        ),
        preemp: bool = True,
        preemp_init = PreempInit(),
        complex_conv_init = GaborInit(
            sample_rate = 16000, min_freq = 60.0, max_freq = 7800.0
        ),
        pooling_init = tf.keras.initializers.Constant(0.4),
        regularizer_fn = None,
        mean_var_norm: bool = True,
        name = 'leaf',
    ):
        super().__init__(name = name)
        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)
        if preemp:
            self._preemp_conv = tf.keras.layers.Conv1D(
                filters = 1,
                kernel_size = 2,
                strides = 1,
                padding = 'SAME',
                use_bias = False,
                input_shape = (None, None, 1),
                kernel_initializer = preemp_init,
                kernel_regularizer = regularizer_fn if learn_filters else None,
                name = 'tfbanks_preemp',
                trainable = learn_filters,
            )

        self._complex_conv = conv1d_cls(
            filters = 2 * n_filters,
            kernel_size = window_size,
            strides = 1,
            padding = 'SAME',
            use_bias = False,
            input_shape = (None, None, 1),
            kernel_initializer = complex_conv_init,
            kernel_regularizer = regularizer_fn if learn_filters else None,
            name = 'tfbanks_complex_conv',
            trainable = learn_filters,
        )

        self._activation = activation
        self._pooling = pooling_cls(
            kernel_size = window_size,
            strides = window_stride,
            padding = 'SAME',
            use_bias = False,
            kernel_initializer = pooling_init,
            kernel_regularizer = regularizer_fn if learn_pooling else None,
            trainable = learn_pooling,
        )

        self._instance_norm = None
        if mean_var_norm:
            self._instance_norm = normalization.InstanceNormalization(
                axis = 2,
                epsilon = 1e-6,
                center = True,
                scale = True,
                beta_initializer = 'zeros',
                gamma_initializer = 'ones',
                name = 'tfbanks_instancenorm',
            )

        self._compress_fn = compression_fn if compression_fn else tf.identity

        self._preemp = preemp

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Computes the Leaf representation of a batch of waveforms.
    Args:
      inputs: input audio of shape (batch_size, num_samples) or (batch_size,
        num_samples, 1).
      training: training mode, controls whether SpecAugment is applied or not.
    Returns:
      Leaf features of shape (batch_size, time_frames, freq_bins).
    """
        # Inputs should be [B, W] or [B, W, C]
        outputs = inputs[:, :, tf.newaxis] if inputs.shape.ndims < 3 else inputs
        if self._preemp:
            outputs = self._preemp_conv(outputs)
        outputs = self._complex_conv(outputs)
        outputs = self._activation(outputs)
        outputs = self._pooling(outputs)
        outputs = tf.maximum(outputs, 1e-5)
        outputs = self._compress_fn(outputs)
        if self._instance_norm is not None:
            outputs = self._instance_norm(outputs)
        return outputs
