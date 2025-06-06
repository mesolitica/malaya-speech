# https://github.com/jik876/hifi-gan/blob/master/config_v2.json
config = {
    'sampling_rate': 22050,
    'hop_size': 256,
    'model_type': 'hifigan_generator',
    'hifigan_generator_params': {
        'out_channels': 1,
        'kernel_size': 7,
        'filters': 128,
        'use_bias': True,
        'upsample_scales': [8, 8, 2, 2],
        'stacks': 3,
        'stack_kernel_size': [3, 7, 11],
        'stack_dilation_rate': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        'use_final_nolinear_activation': True,
        'is_weight_norm': False,
    },
    'hifigan_discriminator_params': {
        'out_channels': 1,
        'period_scales': [2, 3, 5, 7, 11],
        'n_layers': 5,
        'kernel_size': 5,
        'strides': 3,
        'filters': 8,
        'filter_scales': 4,
        'max_filters': 512,
        'is_weight_norm': False,
    },
    'melgan_discriminator_params': {
        'out_channels': 1,
        'scales': 3,
        'downsample_pooling': 'AveragePooling1D',
        'downsample_pooling_params': {'pool_size': 4, 'strides': 2},
        'kernel_sizes': [5, 3],
        'filters': 16,
        'max_downsample_filters': 512,
        'downsample_scales': [4, 4, 4, 4],
        'nonlinear_activation': 'LeakyReLU',
        'nonlinear_activation_params': {'alpha': 0.2},
        'is_weight_norm': False,
    },
    'stft_loss_params': {
        'fft_lengths': [1024, 2048, 512],
        'frame_steps': [120, 240, 50],
        'frame_lengths': [600, 1200, 240],
    },
    'lambda_feat_match': 10.0,
    'lambda_adv': 4.0,
    'batch_size': 16,
    'batch_max_steps': 8192,
    'batch_max_steps_valid': 81920,
    'remove_short_samples': True,
    'allow_cache': True,
    'is_shuffle': True,
}

# https://github.com/jik876/hifi-gan/blob/master/config_v3.json
config_v2 = {
    'sampling_rate': 22050,
    'hop_size': 256,
    'model_type': 'hifigan_generator',
    'hifigan_generator_params': {
        'out_channels': 1,
        'kernel_size': 7,
        'filters': 256,
        'use_bias': True,
        'upsample_scales': [8, 8, 4],
        'stacks': 3,
        'stack_kernel_size': [3, 5, 7],
        'stack_dilation_rate': [[1, 2], [2, 6], [3, 12]],
        'use_final_nolinear_activation': True,
        'is_weight_norm': False,
    },
    'hifigan_discriminator_params': {
        'out_channels': 1,
        'period_scales': [2, 3, 5, 7, 11],
        'n_layers': 5,
        'kernel_size': 5,
        'strides': 3,
        'filters': 8,
        'filter_scales': 4,
        'max_filters': 512,
        'is_weight_norm': False,
    },
    'melgan_discriminator_params': {
        'out_channels': 1,
        'scales': 3,
        'downsample_pooling': 'AveragePooling1D',
        'downsample_pooling_params': {'pool_size': 4, 'strides': 2},
        'kernel_sizes': [5, 3],
        'filters': 16,
        'max_downsample_filters': 512,
        'downsample_scales': [4, 4, 4, 4],
        'nonlinear_activation': 'LeakyReLU',
        'nonlinear_activation_params': {'alpha': 0.2},
        'is_weight_norm': False,
    },
    'stft_loss_params': {
        'fft_lengths': [1024, 2048, 512],
        'frame_steps': [120, 240, 50],
        'frame_lengths': [600, 1200, 240],
    },
    'lambda_feat_match': 10.0,
    'lambda_adv': 4.0,
    'batch_size': 16,
    'batch_max_steps': 8192,
    'batch_max_steps_valid': 81920,
    'remove_short_samples': True,
    'allow_cache': True,
    'is_shuffle': True,
}

# https://github.com/jik876/hifi-gan/blob/master/config_v1.json
config_v3 = {
    'sampling_rate': 22050,
    'hop_size': 256,
    'model_type': 'hifigan_generator',
    'hifigan_generator_params': {
        'out_channels': 1,
        'kernel_size': 7,
        'filters': 512,
        'use_bias': True,
        'upsample_scales': [8, 8, 2, 2],
        'stacks': 3,
        'stack_kernel_size': [3, 7, 11],
        'stack_dilation_rate': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        'use_final_nolinear_activation': True,
        'is_weight_norm': False,
    },
    'hifigan_discriminator_params': {
        'out_channels': 1,
        'period_scales': [2, 3, 5, 7, 11],
        'n_layers': 5,
        'kernel_size': 5,
        'strides': 3,
        'filters': 8,
        'filter_scales': 4,
        'max_filters': 512,
        'is_weight_norm': False,
    },
    'melgan_discriminator_params': {
        'out_channels': 1,
        'scales': 3,
        'downsample_pooling': 'AveragePooling1D',
        'downsample_pooling_params': {'pool_size': 4, 'strides': 2},
        'kernel_sizes': [5, 3],
        'filters': 16,
        'max_downsample_filters': 512,
        'downsample_scales': [4, 4, 4, 4],
        'nonlinear_activation': 'LeakyReLU',
        'nonlinear_activation_params': {'alpha': 0.2},
        'is_weight_norm': False,
    },
    'stft_loss_params': {
        'fft_lengths': [1024, 2048, 512],
        'frame_steps': [120, 240, 50],
        'frame_lengths': [600, 1200, 240],
    },
    'lambda_feat_match': 10.0,
    'lambda_adv': 4.0,
    'batch_size': 16,
    'batch_max_steps': 8192,
    'batch_max_steps_valid': 81920,
    'remove_short_samples': True,
    'allow_cache': True,
    'is_shuffle': True,
}


# for universal, scale up `filters` based on `config_v2`
config_v4 = {
    'sampling_rate': 22050,
    'hop_size': 256,
    'model_type': 'hifigan_generator',
    'hifigan_generator_params': {
        'out_channels': 1,
        'kernel_size': 7,
        'filters': 512,
        'use_bias': True,
        'upsample_scales': [8, 8, 2, 2],
        'stacks': 3,
        'stack_kernel_size': [3, 7, 11],
        'stack_dilation_rate': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        'use_final_nolinear_activation': True,
        'is_weight_norm': False,
    },
    'hifigan_discriminator_params': {
        'out_channels': 1,
        'period_scales': [2, 3, 5, 7, 11],
        'n_layers': 5,
        'kernel_size': 5,
        'strides': 3,
        'filters': 8,
        'filter_scales': 4,
        'max_filters': 512,
        'is_weight_norm': False,
    },
    'melgan_discriminator_params': {
        'out_channels': 1,
        'scales': 3,
        'downsample_pooling': 'AveragePooling1D',
        'downsample_pooling_params': {'pool_size': 4, 'strides': 2},
        'kernel_sizes': [5, 3],
        'filters': 16,
        'max_downsample_filters': 512,
        'downsample_scales': [4, 4, 4, 4],
        'nonlinear_activation': 'LeakyReLU',
        'nonlinear_activation_params': {'alpha': 0.2},
        'is_weight_norm': False,
    },
    'stft_loss_params': {
        'fft_lengths': [1024, 2048, 512],
        'frame_steps': [120, 240, 50],
        'frame_lengths': [600, 1200, 240],
    },
    'lambda_feat_match': 10.0,
    'lambda_adv': 4.0,
    'batch_size': 16,
    'batch_max_steps': 8192,
    'batch_max_steps_valid': 81920,
    'remove_short_samples': True,
    'allow_cache': True,
    'is_shuffle': True,
}
