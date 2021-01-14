from tensorflow.python.keras.layers import Add, Conv1D, Input, Lambda


def res_block(x_in, filters, scaling):
    x = Conv1D(filters, 3, padding = 'same', activation = 'relu')(x_in)
    x = Conv1D(filters, 3, padding = 'same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample1(x, factor, **kwargs):
        x = Conv1D(num_filters, kernel_size = 3, padding = 'same', **kwargs)(x)
        last = int(x.shape[-1])
        return tf.reshape(x, (tf.shape(x)[0], -1, last // factor))

    if scale == 2:
        x = upsample_1(x, 2, name = 'conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name = 'conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name = 'conv2d_1_scale_2')
        x = upsample_1(x, 2, name = 'conv2d_2_scale_2')

    return x


class Model:
    def __init__(
        self,
        inputs,
        scale = 4,
        num_filters = 256,
        num_res_blocks = 16,
        res_block_scaling = None,
    ):
        x = inputs

        x = b = Conv1D(num_filters, 3, padding = 'same')(x)
        for i in range(num_res_blocks):
            b = res_block(b, num_filters, res_block_scaling)
        b = Conv1D(num_filters, 3, padding = 'same')(b)
        x = Add()([x, b])

        x = upsample(x, scale, num_filters)
        self.logits = Conv1D(1, 3, padding = 'same', activation = 'tanh')(x)
