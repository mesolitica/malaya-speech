# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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
"""UnivNet Config object."""


class GeneratorConfig(object):
    """Initialize UnivNet Generator Config."""

    def __init__(
        self,
        noise_dim=64,
        channel_size=32,
        n_mel_channels=80,
        dilations=[1, 3, 9, 27],
        strides=[8, 8, 4],
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        initializer_seed=42,
        **kwargs,
    ):
        """Init parameters for UnivNet Generator model."""
        self.noise_dim = noise_dim
        self.channel_size = channel_size
        self.n_mel_channels = n_mel_channels
        self.dilations = dilations
        self.strides = strides
        self.kpnet_hidden_channels = kpnet_hidden_channels
        self.kpnet_conv_size = kpnet_conv_size
        self.initializer_seed = initializer_seed
        self.hop_length = 256


class WaveFormDiscriminatorConfig(object):
    """Initialize MelGAN Discriminator Config."""

    def __init__(
        self,
        out_channels=1,
        scales=3,
        downsample_pooling='AveragePooling1D',
        downsample_pooling_params={'pool_size': 4, 'strides': 2},
        kernel_sizes=[5, 3],
        filters=16,
        max_downsample_filters=1024,
        use_bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'alpha': 0.2},
        padding_type='REFLECT',
        is_weight_norm=True,
        initializer_seed=42,
        **kwargs
    ):
        """Init parameters for MelGAN Discriminator model."""
        self.out_channels = out_channels
        self.scales = scales
        self.downsample_pooling = downsample_pooling
        self.downsample_pooling_params = downsample_pooling_params
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.max_downsample_filters = max_downsample_filters
        self.use_bias = use_bias
        self.downsample_scales = downsample_scales
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.padding_type = padding_type
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed


class STFTDiscriminatorConfig(object):
    """Initialize MelGAN Discriminator Config."""

    def __init__(
        self,
        fft_length=1024,
        frame_length=1024,
        frame_step=256,
        out_channels=1,
        scales=3,
        downsample_pooling='AveragePooling1D',
        downsample_pooling_params={'pool_size': 4, 'strides': 2},
        kernel_sizes=[5, 3],
        filters=16,
        max_downsample_filters=1024,
        use_bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'alpha': 0.2},
        padding_type='REFLECT',
        is_weight_norm=True,
        initializer_seed=42,
        **kwargs
    ):
        """Init parameters for MelGAN Discriminator model."""
        self.fft_length = fft_length
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.out_channels = out_channels
        self.scales = scales
        self.downsample_pooling = downsample_pooling
        self.downsample_pooling_params = downsample_pooling_params
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.max_downsample_filters = max_downsample_filters
        self.use_bias = use_bias
        self.downsample_scales = downsample_scales
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.padding_type = padding_type
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed
