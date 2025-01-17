# https://github.com/kssteven418/Squeezeformer/tree/main/examples/squeezeformer/configs

xs_encoder_config = {'encoder_subsampling': {'type': 'conv2d',
                                             'filters': 144,
                                             'kernel_size': 3,
                                             'strides': 2},
                     'encoder_dmodel': 144,
                     'encoder_num_blocks': 16,
                     'encoder_head_size': 36,
                     'encoder_num_heads': 4,
                     'encoder_mha_type': 'relmha',
                     'encoder_kernel_size': 31,
                     'encoder_fc_factor': 1.0,
                     'encoder_dropout': 0.1,
                     'encoder_time_reduce_idx': [7],
                     'encoder_time_recover_idx': [15],
                     'encoder_conv_use_glu': False,
                     'encoder_ds_subsample': True,
                     'encoder_no_post_ln': True,
                     'encoder_adaptive_scale': True,
                     'encoder_fixed_arch': ['M', 's', 'C', 's']}

s_encoder_config = {'encoder_subsampling': {'type': 'conv2d',
                                            'filters': 196,
                                            'kernel_size': 3,
                                            'strides': 2},
                    'encoder_dmodel': 196,
                    'encoder_num_blocks': 18,
                    'encoder_head_size': 49,
                    'encoder_num_heads': 4,
                    'encoder_mha_type': 'relmha',
                    'encoder_kernel_size': 31,
                    'encoder_fc_factor': 1.0,
                    'encoder_dropout': 0.1,
                    'encoder_time_reduce_idx': [8],
                    'encoder_time_recover_idx': [17],
                    'encoder_conv_use_glu': False,
                    'encoder_ds_subsample': True,
                    'encoder_no_post_ln': True,
                    'encoder_adaptive_scale': True,
                    'encoder_fixed_arch': ['M', 's', 'C', 's']}

sm_encoder_config = {'encoder_subsampling': {'type': 'conv2d',
                                             'filters': 256,
                                             'kernel_size': 3,
                                             'strides': 2},
                     'encoder_dmodel': 256,
                     'encoder_num_blocks': 16,
                     'encoder_head_size': 64,
                     'encoder_num_heads': 4,
                     'encoder_mha_type': 'relmha',
                     'encoder_kernel_size': 31,
                     'encoder_fc_factor': 1.0,
                     'encoder_dropout': 0.1,
                     'encoder_time_reduce_idx': [7],
                     'encoder_time_recover_idx': [15],
                     'encoder_conv_use_glu': False,
                     'encoder_ds_subsample': True,
                     'encoder_no_post_ln': True,
                     'encoder_adaptive_scale': True,
                     'encoder_fixed_arch': ['M', 's', 'C', 's']}

m_encoder_config = {'encoder_subsampling': {'type': 'conv2d',
                                            'filters': 324,
                                            'kernel_size': 3,
                                            'strides': 2},
                    'encoder_dmodel': 324,
                    'encoder_num_blocks': 20,
                    'encoder_head_size': 81,
                    'encoder_num_heads': 4,
                    'encoder_mha_type': 'relmha',
                    'encoder_kernel_size': 31,
                    'encoder_fc_factor': 1.0,
                    'encoder_dropout': 0.1,
                    'encoder_time_reduce_idx': [9],
                    'encoder_time_recover_idx': [19],
                    'encoder_conv_use_glu': False,
                    'encoder_ds_subsample': True,
                    'encoder_no_post_ln': True,
                    'encoder_adaptive_scale': True,
                    'encoder_fixed_arch': ['M', 's', 'C', 's']}

ml_encoder_config = {'encoder_subsampling': {'type': 'conv2d',
                                             'filters': 512,
                                             'kernel_size': 3,
                                             'strides': 2},
                     'encoder_dmodel': 512,
                     'encoder_num_blocks': 18,
                     'encoder_head_size': 64,
                     'encoder_num_heads': 8,
                     'encoder_mha_type': 'relmha',
                     'encoder_kernel_size': 31,
                     'encoder_fc_factor': 1.0,
                     'encoder_dropout': 0.1,
                     'encoder_time_reduce_idx': [8],
                     'encoder_time_recover_idx': [17],
                     'encoder_conv_use_glu': False,
                     'encoder_ds_subsample': True,
                     'encoder_no_post_ln': True,
                     'encoder_adaptive_scale': True,
                     'encoder_fixed_arch': ['M', 's', 'C', 's']}

l_encoder_config = {'encoder_subsampling': {'type': 'conv2d',
                                            'filters': 640,
                                            'kernel_size': 3,
                                            'strides': 2},
                    'encoder_dmodel': 640,
                    'encoder_num_blocks': 22,
                    'encoder_head_size': 80,
                    'encoder_num_heads': 8,
                    'encoder_mha_type': 'relmha',
                    'encoder_kernel_size': 31,
                    'encoder_fc_factor': 1.0,
                    'encoder_dropout': 0.1,
                    'encoder_time_reduce_idx': [10],
                    'encoder_time_recover_idx': [21],
                    'encoder_conv_use_glu': False,
                    'encoder_ds_subsample': True,
                    'encoder_no_post_ln': True,
                    'encoder_adaptive_scale': True,
                    'encoder_fixed_arch': ['M', 's', 'C', 's']}
