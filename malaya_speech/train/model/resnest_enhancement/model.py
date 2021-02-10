import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation
from . import layer as lay


class UpSamplingLayer:
    def __init__(self, channel_out, kernel_size = 5, stride = 1):
        self.seq = tf.keras.Sequential()
        self.seq.add(
            tf.keras.layers.Conv1D(
                channel_out,
                kernel_size = kernel_size,
                strides = stride,
                padding = 'SAME',
                dilation_rate = 1,
            )
        )
        self.seq.add(BatchNormalization(axis = -1))
        self.seq.add(LeakyReLU(0.2))

    def __call__(self, x, training = True):
        return self.seq(x, training = training)


class Model:
    def __init__(
        self,
        inputs,
        training = True,
        ksize = 3,
        radix = 4,
        kpaths = 4,
        n_layers = 12,
        channels_interval = 24,
        logging = True,
    ):
        self.customlayers = lay.Layers()
        self.n_layers = n_layers
        self.channels_interval = channels_interval
        out_channels = [
            i * self.channels_interval for i in range(1, self.n_layers + 1)
        ]
        self.middle = tf.keras.Sequential()
        self.middle.add(
            tf.keras.layers.Conv1D(
                self.n_layers * self.channels_interval,
                kernel_size = 15,
                strides = 1,
                padding = 'SAME',
                dilation_rate = 1,
            )
        )
        self.middle.add(BatchNormalization(axis = -1))
        self.middle.add(LeakyReLU(0.2))

        decoder_out_channels_list = out_channels[::-1]

        self.decoder = []
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(channel_out = decoder_out_channels_list[i])
            )
        self.out = tf.keras.Sequential()
        self.out.add(
            tf.keras.layers.Conv1D(
                1,
                kernel_size = 1,
                strides = 1,
                padding = 'SAME',
                dilation_rate = 1,
            )
        )
        self.out.add(Activation('tanh'))

        tmp = []
        o = inputs

        for i in range(self.n_layers):
            o = self.residual_S(
                o,
                ksize = ksize,
                inchannel = int(o.shape[-1]),
                outchannel = out_channels[i],
                radix = radix,
                kpaths = kpaths,
                name = f'down_{i}',
                logging = logging,
                training = training,
            )
            tmp.append(o)
            o = o[:, ::2]
            if logging:
                print(o)

        o = self.middle(o, training = training)
        if logging:
            print(o)

        for i in range(self.n_layers):
            o = tf.image.resize(
                o, [tf.shape(o)[0], tf.shape(o)[1] * 2], method = 'nearest'
            )
            o = tf.concat([o, tmp[self.n_layers - i - 1]], axis = 2)
            o = self.decoder[i](o, training = training)
            if logging:
                print(o)

        if logging:
            print(o, inputs)
        o = tf.concat([o, inputs], axis = 2)
        o = self.out(o, training = training)
        self.logits = o

    def residual_S(
        self,
        input,
        ksize,
        inchannel,
        outchannel,
        radix,
        kpaths,
        name = '',
        logging = False,
        training = True,
    ):
        convtmp_1 = self.customlayers.conv1d(
            input,
            self.customlayers.get_weight(
                vshape = [ksize, inchannel, outchannel], name = '%s_1' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_1bn = BatchNormalization(axis = -1)(
            convtmp_1, training = training
        )
        convtmp_1act = self.customlayers.elu(convtmp_1bn)
        convtmp_2 = self.customlayers.conv1d(
            convtmp_1act,
            self.customlayers.get_weight(
                vshape = [ksize, outchannel, outchannel], name = '%s_2' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_2bn = BatchNormalization(axis = -1)(
            convtmp_2, training = training
        )

        convtmp_2act = self.customlayers.elu(convtmp_2bn)

        concats_1 = None
        for idx_k in range(kpaths):
            cardinal = self.cardinal(
                convtmp_2act,
                ksize,
                outchannel,
                outchannel,
                radix,
                kpaths,
                name = '%s_car_k%d' % (name, idx_k),
                training = training,
            )
            if idx_k == 0:
                concats_1 = cardinal
            else:
                concats_1 = tf.concat([concats_1, cardinal], axis = 2)
        concats_2 = self.customlayers.conv1d(
            concats_1,
            self.customlayers.get_weight(
                vshape = [1, outchannel, outchannel], name = '%s_cc' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        concats_2 = concats_2 + convtmp_2act

        if input.shape[-1] != concats_2.shape[-1]:
            convtmp_sc = self.customlayers.conv1d(
                input,
                self.customlayers.get_weight(
                    vshape = [1, inchannel, outchannel], name = '%s_sc' % (name)
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            convtmp_scbn = BatchNormalization(axis = -1)(
                convtmp_sc, training = training
            )
            convtmp_scact = self.customlayers.elu(convtmp_scbn)
            input = convtmp_scact

        output = input + concats_2
        if logging:
            print(name, output.shape)
        return output

    def cardinal(
        self,
        input,
        ksize,
        inchannel,
        outchannel,
        radix,
        kpaths,
        name = '',
        logging = False,
        training = True,
    ):

        if logging:
            print('cardinal')
        outchannel_cv11 = int(outchannel / radix / kpaths)
        outchannel_cvkk = int(outchannel / kpaths)

        inputs = []
        for idx_r in range(radix):
            conv1 = self.customlayers.conv1d(
                input,
                self.customlayers.get_weight(
                    vshape = [1, inchannel, outchannel_cv11],
                    name = '%s1_r%d' % (name, idx_r),
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            conv1_bn = BatchNormalization(axis = -1)(conv1, training = training)
            conv1_act = self.customlayers.elu(conv1_bn)

            conv2 = self.customlayers.conv1d(
                conv1_act,
                self.customlayers.get_weight(
                    vshape = [ksize, outchannel_cv11, outchannel_cvkk],
                    name = '%s2_r%d' % (name, idx_r),
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            conv2_bn = BatchNormalization(axis = -1)(conv2, training = training)
            conv2_act = self.customlayers.elu(conv2_bn)
            inputs.append(conv2_act)

        return self.split_attention(
            inputs,
            outchannel_cvkk,
            name = '%s_att' % (name),
            training = training,
        )

    def split_attention(
        self, inputs, inchannel, name = '', verbose = False, training = True
    ):

        if verbose:
            print('split attention')
        radix = len(inputs)
        input_holder = None
        for idx_i, input in enumerate(inputs):
            if idx_i == 0:
                input_holder = input
            else:
                input_holder += input
        ga_pool = tf.math.reduce_mean(input_holder, axis = (1))
        ga_pool = tf.expand_dims(ga_pool, axis = 1)

        dense1 = self.customlayers.conv1d(
            ga_pool,
            self.customlayers.get_weight(
                vshape = [1, inchannel, inchannel], name = '%s1' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        dense1_bn = BatchNormalization(axis = -1)(dense1, training = training)
        dense1_act = self.customlayers.elu(dense1_bn)

        output_holder = None
        for idx_r in range(radix):
            dense2 = self.customlayers.conv1d(
                dense1_act,
                self.customlayers.get_weight(
                    vshape = [1, inchannel, inchannel],
                    name = '%s2_r%d' % (name, idx_r),
                ),
                stride_size = 1,
                padding = 'SAME',
            )
            if radix == 1:
                r_softmax = self.customlayers.sigmoid(dense2)
            elif radix > 1:
                r_softmax = self.customlayers.softmax(dense2)

            if idx_r == 0:
                output_holder = inputs[idx_r] * r_softmax
            else:
                output_holder += inputs[idx_r] * r_softmax

        return output_holder
