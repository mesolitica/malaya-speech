import tensorflow as tf


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, dims, **kwargs):
        super(ResBlock, self).__init__(name = 'Conv1DTranspose', **kwargs)
        self.conv1 = tf.layers.Conv1D(dims, kernel_size = 1, bias = False)
        self.conv2 = tf.layers.Conv1D(dims, kernel_size = 1, bias = False)
        self.batch_norm1 = tf.layers.BatchNormalization()
        self.batch_norm2 = tf.layers.BatchNormalization()

    def call(self, x, training = False):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x, training = training)
        return x + residual


class MelResNet(tf.keras.layers.Layer):
    def __init__(
        self, res_blocks, in_dims, compute_dims, res_out_dims, pad, **kwargs
    ):
        super(MelResNet, self).__init__(name = 'MelResNet', **kwargs)
        k_size = pad * 2 + 1
        self.conv_in = tf.layers.Conv1D(
            compute_dims, kernel_size = k_size, bias = False
        )
        self.batch_norm = tf.layers.BatchNormalization()
        self.layers = tf.keras.Sequential()
        for i in range(res_blocks):
            self.layers.add(ResBlock(compute_dims))
        self.conv_out = tf.layers.Conv1D(res_out_dims, kernel_size = 1)

    def call(self, x, training = False):
        x = self.conv_in(x)
        x = self.batch_norm(x, training = training)
        x = tf.nn.relu(x)
        x = self.layers(x)
        x = self.conv_out(x)
        return x


class Stretch2d(tf.keras.layers.Layer):
    def __init__(self, x_scale, y_scale, **kwargs):
        super(Stretch2d, self).__init__(name = 'Stretch2d', **kwargs)
        self.x_scale = x_scale
        self.y_scale = y_scale

    def call(self, x):
        shape = tf.shape(x)
        b = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        x = tf.expand_dims(tf.expand_dims(x, 2), 4)
        x = tf.tile(x, [1, 1, self.y_scale, 1, self.x_scale, 1])
        return tf.reshape(x, [b, h * self.y_scale, w * self.x_scale, c])


class UpsampleNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        feat_dims,
        upsample_scales,
        compute_dims,
        res_blocks,
        res_out_dims,
        pad,
    ):
        super(UpsampleNetwork, self).__init__(
            name = 'UpsampleNetwork', **kwargs
        )
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(
            res_blocks, feat_dims, compute_dims, res_out_dims, pad
        )
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = tf.keras.Sequential()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = tf.keras.layers.Conv2D(
                1, kernel_size = k_size, padding = 'SAME', bias = False
            )
            self.up_layers.add(stretch)
            self.up_layers.add(conv)


class Model:
    def __init__(self, input_tensor):
        pass
