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
from .subsampling import Conv2dSubsampling
from .time_reduction import TimeReductionLayer
from .multihead_attention import MultiHeadAttention, RelPositionMultiHeadAttention
# from .positional_encoding import PositionalEncoding
from ..conformer.layer import PositionalEncoding
from ..utils import shape_list
import logging

logger = logging.getLogger(__name__)


class FFModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        adaptive_scale=False,
        ff_expansion_rate=4,
        name="ff_module",
        **kwargs,
    ):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        logger.info(f"fc factor set as {self.fc_factor}")

        self.adaptive_scale = adaptive_scale
        if not adaptive_scale:
            logger.info("No scaling, use preLN")
            self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")
        else:
            logger.info("Use scaling, no preLN")
            self.scale = tf.Variable([1.] * input_dim, trainable=True, name=f'{name}_scale')
            self.bias = tf.Variable([0.] * input_dim, trainable=True, name=f'{name}_bias')
        ffn1_max = input_dim ** -0.5
        ffn2_max = (ff_expansion_rate * input_dim) ** -0.5
        self.ffn1 = tf.keras.layers.Dense(
            ff_expansion_rate * input_dim, name=f"{name}_dense_1",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-ffn1_max, maxval=ffn1_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-ffn1_max, maxval=ffn1_max),
        )
        self.act = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_act")
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim, name=f"{name}_dense_2",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-ffn2_max, maxval=ffn2_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-ffn2_max, maxval=ffn2_max),
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, **kwargs):
        if not self.adaptive_scale:
            outputs = self.ln(inputs, training=training)
        else:
            scale = tf.reshape(self.scale, (1, 1, -1))
            bias = tf.reshape(self.bias, (1, 1, -1))
            outputs = inputs * scale + bias
        outputs = self.ffn1(outputs, training=training)
        outputs = self.act(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs


class MHSAModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size,
        num_heads,
        dropout=0.0,
        mha_type="relmha",
        adaptive_scale=False,
        name="mhsa_module",
        **kwargs,
    ):
        super(MHSAModule, self).__init__(name=name, **kwargs)

        self.adaptive_scale = adaptive_scale
        input_dim = num_heads * head_size
        if not adaptive_scale:
            logger.info("No scaling, use preLN")
            self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")
        else:
            logger.info("Use scaling, no preLN")
            self.scale = tf.Variable([1.] * input_dim, trainable=True, name=f'{name}_scale')
            self.bias = tf.Variable([0.] * input_dim, trainable=True, name=f'{name}_bias')

        if mha_type == "relmha":
            self.mha = RelPositionMultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size, num_heads=num_heads,
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size, num_heads=num_heads,
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(self, inputs, training=False, mask=None, pos=False, **kwargs):
        if pos is False:
            inputs, pos = inputs  # pos is positional encoding

        if not self.adaptive_scale:
            outputs = self.ln(inputs, training=training)
        else:
            scale = tf.reshape(self.scale, (1, 1, -1))
            bias = tf.reshape(self.bias, (1, 1, -1))
            outputs = inputs * scale + bias
        if self.mha_type == "relmha":
            outputs = self.mha([outputs, outputs, outputs, pos], training=training, mask=mask)
        else:
            outputs = outputs + pos
            outputs = self.mha([outputs, outputs, outputs], training=training, mask=mask)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs


class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        kernel_size=31,
        dropout=0.0,
        depth_multiplier=1,
        conv_expansion_rate=2,
        conv_use_glu=False,
        adaptive_scale=False,
        name="conv_module",
        **kwargs,
    ):
        super(ConvModule, self).__init__(name=name, **kwargs)

        self.adaptive_scale = adaptive_scale
        if not adaptive_scale:
            logger.info("No scaling, use preLN")
            self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")
        else:
            logger.info("Use scaling, no preLN")
            self.scale = tf.Variable([1.] * input_dim, trainable=True, name=f'{name}_scale')
            self.bias = tf.Variable([0.] * input_dim, trainable=True, name=f'{name}_bias')
        pw1_max = input_dim ** -0.5
        dw_max = kernel_size ** -0.5
        pw2_max = input_dim ** -0.5
        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=conv_expansion_rate * input_dim, kernel_size=1, strides=1,
            padding="valid", name=f"{name}_pw_conv_1",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-pw1_max, maxval=pw1_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-pw1_max, maxval=pw1_max),
        )
        if conv_use_glu:
            logger.info("Using GLU for Conv")
            self.act1 = GLU(name=f"{name}_act_1")
        else:
            logger.info("Replace GLU with swish for Conv")
            self.act1 = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_act_1")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1), strides=1,
            padding="same", name=f"{name}_dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-dw_max, maxval=dw_max),
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f'{name}_bn',
            momentum=0.985,
        )
        self.act2 = tf.keras.layers.Activation(tf.nn.swish, name=f"{name}_act_2")
        self.pw_conv_2 = tf.keras.layers.Conv2D(
            filters=input_dim, kernel_size=1, strides=1,
            padding="valid", name=f"{name}_pw_conv_2",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-pw2_max, maxval=pw2_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-pw2_max, maxval=pw2_max),
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, pad_mask=None, **kwargs):
        if not self.adaptive_scale:
            outputs = self.ln(inputs, training=training)
        else:
            scale = tf.reshape(self.scale, (1, 1, -1))
            bias = tf.reshape(self.bias, (1, 1, -1))
            outputs = inputs * scale + bias
        B, T, E = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, 1, E])
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.act1(outputs)
        pad_mask = tf.expand_dims(tf.expand_dims(pad_mask, -1), -1)
        outputs = outputs * tf.cast(pad_mask, "float32")
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.act2(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs


class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return inputs


class MHSAFFModule(tf.keras.layers.Layer):
    '''
    Wrapper class for a MHSA layer followed by a FF layer
    '''

    def __init__(
        self,
        input_dim,
        head_size,
        num_heads,
        dropout=0.0,
        mha_type="relmha",
        fc_factor=0.5,
        ff_expansion_rate=4,
        adaptive_scale=False,
        name="mhsaff_module",
        **kwargs,
    ):
        super(MHSAFFModule, self).__init__(name=name, **kwargs)
        assert input_dim == head_size * num_heads
        self.mhsa = MHSAModule(
            mha_type=mha_type,
            head_size=head_size,
            num_heads=num_heads,
            adaptive_scale=adaptive_scale,
            dropout=dropout,
            name=f"{name}_mhsa",
        )
        self.ln_mid = tf.keras.layers.LayerNormalization(name=f"{name}_ln_mid")
        self.ff = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            ff_expansion_rate=ff_expansion_rate,
            adaptive_scale=adaptive_scale,
            name=f"{name}_ff",
        )
        self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")

    def call(self, inputs, training=False, *args, **kwargs):
        outputs = self.mhsa(inputs, training=training, *args, **kwargs)
        outputs = self.ln_mid(outputs, training=training)
        outputs = self.ff(outputs, training=training, *args, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs


class ConvFFModule(tf.keras.layers.Layer):
    '''
    Wrapper class for a Conv layer followed by a FF layer
    '''

    def __init__(
        self,
        input_dim,
        kernel_size=31,
        dropout=0.0,
        conv_expansion_rate=2,
        conv_use_glu=False,
        fc_factor=0.5,
        ff_expansion_rate=4,
        adaptive_scale=False,
        name="convff_module",
        **kwargs,
    ):
        super(ConvFFModule, self).__init__(name=name, **kwargs)
        self.conv = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            conv_expansion_rate=conv_expansion_rate,
            dropout=dropout,
            conv_use_glu=conv_use_glu,
            adaptive_scale=adaptive_scale,
            name=f"{name}_conv",
        )
        self.ln_mid = tf.keras.layers.LayerNormalization(name=f"{name}_ln_mid")
        self.ff = FFModule(
            input_dim=input_dim, dropout=dropout,
            fc_factor=fc_factor,
            ff_expansion_rate=ff_expansion_rate,
            adaptive_scale=adaptive_scale,
            name=f"{name}_ff",
        )
        self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")

    def call(self, inputs, training=False, *args, **kwargs):
        outputs = self.conv(inputs, training=training, *args, **kwargs)
        outputs = self.ln_mid(outputs, training=training)
        outputs = self.ff(outputs, training=training, *args, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
        fc_factor=0.5,
        head_size=36,
        num_heads=4,
        mha_type="relmha",
        kernel_size=31,
        name="conformer_block",
        fixed_arch=None,
        conv_use_glu=False,
        no_post_ln=False,
        adaptive_scale=False,
        **kwargs,
    ):
        assert input_dim == num_heads * head_size
        super(ConformerBlock, self).__init__(name=name, **kwargs)

        def get_fixed_arch(arch_type, name):
            logger.info(f'layer type: {arch_type}')
            if arch_type == 'f':
                return FFModule(
                    input_dim=input_dim,
                    dropout=dropout,
                    fc_factor=fc_factor,
                    adaptive_scale=adaptive_scale,
                    name=name,
                )
            elif arch_type == 'm':
                return MHSAModule(
                    mha_type=mha_type,
                    head_size=head_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    adaptive_scale=adaptive_scale,
                    name=name,
                )
            elif arch_type == 'c':
                return ConvModule(
                    input_dim=input_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    conv_use_glu=conv_use_glu,
                    adaptive_scale=adaptive_scale,
                    name=name,
                )
            elif arch_type == 'M':
                return MHSAFFModule(
                    mha_type=mha_type,
                    head_size=head_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    input_dim=input_dim,
                    fc_factor=fc_factor,
                    adaptive_scale=adaptive_scale,
                    name=name,
                )
            elif arch_type == 'C':
                return ConvFFModule(
                    input_dim=input_dim,
                    kernel_size=kernel_size,
                    conv_use_glu=conv_use_glu,
                    dropout=dropout,
                    fc_factor=fc_factor,
                    adaptive_scale=adaptive_scale,
                    name=name,
                )
            elif arch_type == 's':
                return IdentityLayer()

            raise ValueError(f"fixed architecture type '{arch_type}' is not defined")

        ####### Layer 1: MHSA ######
        print(fixed_arch)

        if fixed_arch is None:
            arch_type = 'm'
        else:
            arch_type = fixed_arch[0]
        self.layer1 = get_fixed_arch(arch_type, name+"_layer1")

        ####### Layer 2: FF ######

        arch_type = 'f' if fixed_arch is None else fixed_arch[1]
        self.layer2 = get_fixed_arch(arch_type, name+"_layer2")

        ####### Layer 3: CONV ######

        arch_type = 'c' if fixed_arch is None else fixed_arch[2]
        self.layer3 = get_fixed_arch(arch_type, name+"_layer3")

        ####### Layer 4: FF ######

        arch_type = 'f' if fixed_arch is None else fixed_arch[3]
        self.layer4 = get_fixed_arch(arch_type, name+"_layer4")

        if not no_post_ln:
            self.ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln")
        else:  # we skip postLN for squeezenet as it has already been applied in MF or CF blocks
            logger.info("Skipping post ln")
            self.ln = None

    def call(self, inputs, training=False, mask=None, pad_mask=None, **kwargs):
        inputs, pos = inputs  # pos is positional encoding
        outputs = self.layer1(inputs, training=training, mask=mask, pos=pos, pad_mask=pad_mask, **kwargs)
        outputs = self.layer2(outputs, training=training, mask=mask, pos=pos, pad_mask=pad_mask, **kwargs)
        outputs = self.layer3(outputs, training=training, mask=mask, pos=pos, pad_mask=pad_mask, **kwargs)
        outputs = self.layer4(outputs, training=training, mask=mask, pos=pos, pad_mask=pad_mask, **kwargs)
        if self.ln is not None:
            outputs = self.ln(outputs, training=training)
        return outputs


class ConformerEncoder(tf.keras.Model):
    def __init__(
        self,
        subsampling,
        dmodel=144,
        num_blocks=16,
        mha_type="relmha",
        head_size=36,
        num_heads=4,
        kernel_size=31,
        fc_factor=0.5,
        dropout=0.0,
        name="conformer_encoder",
        fixed_arch=None,
        conv_use_glu=None,
        time_reduce_idx=None,
        time_recover_idx=None,
        no_post_ln=False,
        ds_subsample=False,
        adaptive_scale=False,
        **kwargs,
    ):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)

        if time_reduce_idx is None:
            self.time_reduce = None
        else:
            if time_recover_idx is None:
                self.time_reduce = 'normal'  # no recovery at the end
            else:
                self.time_reduce = 'recover'  # recovery at the end
                assert len(time_reduce_idx) == len(time_recover_idx)
            self.reduce_idx = time_reduce_idx
            self.recover_idx = time_recover_idx
            self.reduce_stride = 2

        self.dmodel = dmodel
        self.xscale = dmodel ** 0.5
        subsampling_name = subsampling.pop("type", "conv2d")
        if subsampling_name == "vgg":
            raise NotImplementedError("VGG subsampling is not supported")
        elif subsampling_name == "conv2d":
            subsampling_class = Conv2dSubsampling
        elif subsampling_name == 'none':
            subsampling_class = None
        else:
            raise ValueError("subsampling must be either  'conv2d' or 'vgg'")

        if subsampling_class:
            self.conv_subsampling = subsampling_class(
                **subsampling, ds=ds_subsample, name=f"{name}_subsampling",
            )
        else:
            self.conv_subsampling = None

        self.pre_ln = tf.keras.layers.LayerNormalization(name=f"{name}_preln")

        self.pe = PositionalEncoding(dmodel, name=f"{name}_pe")

        linear_max = 5120 ** -0.5  # TODO: parameterize this later
        self.linear = tf.keras.layers.Dense(
            dmodel, name=f"{name}_linear",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-linear_max, maxval=linear_max),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-linear_max, maxval=linear_max),
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

        self.conformer_blocks = []

        recover_dmodels = []
        recover_head_sizes = []
        self.pe_time_reduction = []
        self.time_reduction_layers = []
        self.time_recover_layers = []

        for i in range(num_blocks):
            logger.info(f"Initialize block {i}")
            if self.time_reduce is not None and i in self.reduce_idx:
                recover_dmodel = dmodel
                recover_dmodels.append(recover_dmodel)  # push dmodel to recover later
                recover_head_sizes.append(head_size)  # push head size to recover later
                logger.info(f"Reducing to dmodel {dmodel}, head_size {head_size}")

                self.time_reduction_layers.append(
                    TimeReductionLayer(
                        recover_dmodel,
                        dmodel,
                        stride=self.reduce_stride,
                        name=f"{name}_timereduce",
                    )
                )
                self.pe_time_reduction.append(PositionalEncoding(dmodel, name=f"{name}_pe2"))

            if self.time_reduce == 'recover' and i in self.recover_idx:
                dmodel = recover_dmodels[-1]  # pop dmodel for recovery
                head_size = recover_head_sizes[-1]  # pop head size for recovery
                logger.info(f"recovering to dmodel {dmodel}, head_size {head_size}")

                self.time_recover_layers.append(tf.keras.layers.Dense(dmodel))
                recover_dmodels = recover_dmodels[:-1]
                recover_head_sizes = recover_head_sizes[:-1]

            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                name=f"{name}_block_{i}",
                fixed_arch=None if fixed_arch is None else fixed_arch[i],
                no_post_ln=no_post_ln,
                conv_use_glu=conv_use_glu,
                adaptive_scale=adaptive_scale,
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, inputs, length, training=False, mask=None, **kwargs):
        # input with shape [B, T, V1, V2]
        if self.conv_subsampling:
            outputs = self.conv_subsampling(inputs, training=training)
        else:
            outputs = inputs
        outputs = self.linear(outputs, training=training)
        padding, kernel_size, stride, num_subsample = 1, 3, 2, 2  # TODO: set these in __init__
        for _ in range(num_subsample):
            length = tf.math.ceil((tf.cast(length, tf.float32) + (2 * padding) -
                                  (kernel_size - 1) - 1) / float(stride) + 1)
        pad_mask = tf.sequence_mask(length, maxlen=tf.shape(outputs)[1])
        mask = tf.expand_dims(pad_mask, 1)
        mask = tf.repeat(mask, repeats=[tf.shape(mask)[-1]], axis=1)
        mask = tf.math.logical_and(tf.transpose(mask, perm=[0, 2, 1]), mask)
        pe = self.pe(outputs)
        outputs = outputs * self.xscale
        outputs = self.do(outputs, training=training)
        pe_org, mask_org = pe, mask

        recover_activations = []
        index = 0  # index to point the queues for pe, recover activations, etc.

        outputs = self.pre_ln(outputs, training=training)
        for i, cblock in enumerate(self.conformer_blocks):
            if self.time_reduce is not None and i in self.reduce_idx:
                recover_activations.append((outputs, mask, pad_mask, pe))
                outputs, mask, pad_mask = self.time_reduction_layers[index](
                    outputs, training=training, mask=mask, pad_mask=pad_mask, **kwargs,
                )
                pe = self.pe_time_reduction[index](outputs)
                index += 1

            if self.time_reduce == 'recover' and i in self.recover_idx:
                index -= 1
                recover_activation, mask, pad_mask, pe = recover_activations[index]
                B, T, E = shape_list(outputs)
                outputs = tf.repeat(outputs, [self.reduce_stride] * T, axis=1)
                B, T, E = shape_list(recover_activation)
                outputs = self.time_recover_layers[index](outputs[:, :T, :], training=training)
                outputs = outputs + recover_activation

            outputs = cblock([outputs, pe], training=training, mask=mask, pad_mask=pad_mask, **kwargs)
        return outputs
