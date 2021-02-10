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
        ksize = 5,
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
            o = self.residual_X(
                o,
                ksize = ksize,
                inchannel = int(o.shape[-1]),
                outchannel = out_channels[i],
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

    def residual_X(
        self,
        input,
        ksize,
        inchannel,
        outchannel,
        name = '',
        logging = False,
        training = True,
    ):
        convtmp_1 = self.customlayers.conv1d(
            input,
            self.customlayers.get_weight(
                vshape = [1, inchannel, inchannel], name = '%s_1' % (name)
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
                vshape = [ksize, inchannel, inchannel], name = '%s_2' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_2bn = BatchNormalization(axis = -1)(
            convtmp_2, training = training
        )
        convtmp_2act = self.customlayers.elu(convtmp_2bn)
        convtmp_3 = self.customlayers.conv1d(
            convtmp_2act,
            self.customlayers.get_weight(
                vshape = [ksize, inchannel, outchannel], name = '%s_3' % (name)
            ),
            stride_size = 1,
            padding = 'SAME',
        )
        convtmp_3bn = BatchNormalization(axis = -1)(
            convtmp_3, training = training
        )
        convtmp_3act = self.customlayers.elu(convtmp_3bn)

        if input.shape[-1] != convtmp_3act.shape[-1]:
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

        output = input + convtmp_3act

        if logging:
            print(name, output.shape)
        return output
