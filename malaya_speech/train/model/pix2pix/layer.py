import tensorflow as tf


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(
        batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'CONSTANT'
    )
    return tf.layers.conv2d(
        padded_input,
        out_channels,
        kernel_size = 4,
        strides = (stride, stride),
        padding = 'valid',
        kernel_initializer = tf.random_normal_initializer(0, 0.02),
    )


def gen_conv(batch_input, out_channels, separable_conv = True):
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(
            batch_input,
            out_channels,
            kernel_size = 4,
            strides = (2, 2),
            padding = 'same',
            depthwise_initializer = initializer,
            pointwise_initializer = initializer,
        )
    else:
        return tf.layers.conv2d(
            batch_input,
            out_channels,
            kernel_size = 4,
            strides = (2, 2),
            padding = 'same',
            kernel_initializer = initializer,
        )


def gen_deconv(batch_input, out_channels, separable_conv = True):
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(
            batch_input,
            [h * 2, w * 2],
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        return tf.layers.separable_conv2d(
            resized_input,
            out_channels,
            kernel_size = 4,
            strides = (1, 1),
            padding = 'same',
            depthwise_initializer = initializer,
            pointwise_initializer = initializer,
        )
    else:
        return tf.layers.conv2d_transpose(
            batch_input,
            out_channels,
            kernel_size = 4,
            strides = (2, 2),
            padding = 'same',
            kernel_initializer = initializer,
        )


def lrelu(x, a):
    with tf.name_scope('lrelu'):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(
        inputs,
        axis = 3,
        epsilon = 1e-5,
        momentum = 0.1,
        training = True,
        gamma_initializer = tf.random_normal_initializer(1.0, 0.02),
    )
