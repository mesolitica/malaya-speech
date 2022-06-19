# Copyright 2022 https://github.com/kssteven418/Squeezeformer
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
from .encoder import ConformerEncoder


class Model(tf.keras.Model):
    def __init__(
            self,
            encoder_subsampling: dict,
            encoder_dmodel: int = 144,
            encoder_num_blocks: int = 16,
            encoder_head_size: int = 36,
            encoder_num_heads: int = 4,
            encoder_mha_type: str = "relmha",
            encoder_kernel_size: int = 32,
            encoder_fc_factor: float = 0.5,
            encoder_dropout: float = 0,
            encoder_time_reduce_idx: list = None,
            encoder_time_recover_idx: list = None,
            encoder_conv_use_glu: bool = False,
            encoder_ds_subsample: bool = False,
            encoder_no_post_ln: bool = False,
            encoder_adaptive_scale: bool = False,
            encoder_fixed_arch: list = None,
            name: str = "conformer",
            **kwargs):
        super(Model, self).__init__(name=name, **kwargs)
        if not isinstance(encoder_fixed_arch[0], list):
            encoder_fixed_arch = [encoder_fixed_arch] * encoder_num_blocks
        self.encoder = ConformerEncoder(
            subsampling=encoder_subsampling,
            dmodel=encoder_dmodel,
            num_blocks=encoder_num_blocks,
            head_size=encoder_head_size,
            num_heads=encoder_num_heads,
            mha_type=encoder_mha_type,
            kernel_size=encoder_kernel_size,
            fc_factor=encoder_fc_factor,
            dropout=encoder_dropout,
            time_reduce_idx=encoder_time_reduce_idx,
            time_recover_idx=encoder_time_recover_idx,
            conv_use_glu=encoder_conv_use_glu,
            ds_subsample=encoder_ds_subsample,
            no_post_ln=encoder_no_post_ln,
            adaptive_scale=encoder_adaptive_scale,
            fixed_arch=encoder_fixed_arch,
            name=f"{name}_encoder",
        )

    def call(self, inputs, inputs_length, training=False, **kwargs):
        logits = self.encoder(inputs, inputs_length, training=training, **kwargs)
        return logits
