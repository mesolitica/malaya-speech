# https://github.com/YeongHyeon/ResNeXt-TF2

import tensorflow as tf


class Layers(object):
    def __init__(self):

        self.name_bank, self.params_trainable = [], []
        self.num_params = 0
        self.initializer_xavier = tf.initializers.glorot_normal()

    def elu(self, inputs):
        return tf.nn.elu(inputs)

    def relu(self, inputs):
        return tf.nn.relu(inputs)

    def sigmoid(self, inputs):
        return tf.nn.sigmoid(inputs)

    def softmax(self, inputs):
        return tf.nn.softmax(inputs)

    def dropout(self, inputs, rate):
        return tf.nn.dropout(inputs, rate = rate)

    def maxpool(self, inputs, pool_size, stride_size):

        return tf.nn.max_pool1d(
            inputs,
            ksize = [1, pool_size, pool_size, 1],
            padding = 'VALID',
            strides = [1, stride_size, stride_size, 1],
        )

    def avgpool(self, inputs, pool_size, stride_size):

        return tf.nn.avg_pool1d(
            inputs,
            ksize = [1, pool_size, pool_size, 1],
            padding = 'VALID',
            strides = [1, stride_size, stride_size, 1],
        )

    def get_weight(self, vshape, transpose = False, bias = True, name = ''):

        try:
            idx_w = self.name_bank.index('%s_w' % (name))
            if bias:
                idx_b = self.name_bank.index('%s_b' % (name))
        except:
            w = tf.Variable(
                self.initializer_xavier(vshape),
                name = '%s_w' % (name),
                trainable = True,
                dtype = tf.float32,
            )
            self.name_bank.append('%s_w' % (name))
            self.params_trainable.append(w)

            tmpparams = 1
            for d in vshape:
                tmpparams *= d
            self.num_params += tmpparams

            if bias:
                if transpose:
                    b = tf.Variable(
                        self.initializer_xavier([vshape[-2]]),
                        name = '%s_b' % (name),
                        trainable = True,
                        dtype = tf.float32,
                    )
                else:
                    b = tf.Variable(
                        self.initializer_xavier([vshape[-1]]),
                        name = '%s_b' % (name),
                        trainable = True,
                        dtype = tf.float32,
                    )
                self.name_bank.append('%s_b' % (name))
                self.params_trainable.append(b)

                self.num_params += vshape[-2]
        else:
            w = self.params_trainable[idx_w]
            if bias:
                b = self.params_trainable[idx_b]

        if bias:
            return w, b
        else:
            return w

    def fullcon(self, inputs, variables):

        [weights, biasis] = variables
        out = tf.matmul(inputs, weights) + biasis

        return out

    def conv1d(self, inputs, variables, stride_size, padding):

        [weights, biasis] = variables
        out = (
            tf.nn.conv1d(
                inputs, weights, stride = [1, stride_size, 1], padding = padding
            )
            + biasis
        )

        return out

    def batch_norm(self, inputs, name = ''):

        # https://arxiv.org/pdf/1502.03167.pdf

        mean = tf.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs)
        var = std ** 2

        try:
            idx_offset = self.name_bank.index('%s_offset' % (name))
            idx_scale = self.name_bank.index('%s_scale' % (name))
        except:
            offset = tf.Variable(
                0,
                name = '%s_offset' % (name),
                trainable = True,
                dtype = tf.float32,
            )
            self.name_bank.append('%s_offset' % (name))
            self.params_trainable.append(offset)
            self.num_params += 1
            scale = tf.Variable(
                1,
                name = '%s_scale' % (name),
                trainable = True,
                dtype = tf.float32,
            )
            self.name_bank.append('%s_scale' % (name))
            self.params_trainable.append(scale)
            self.num_params += 1
        else:
            offset = self.params_trainable[idx_offset]
            scale = self.params_trainable[idx_scale]

        offset  # zero
        scale  # one
        out = tf.nn.batch_normalization(
            x = inputs,
            mean = mean,
            variance = var,
            offset = offset,
            scale = scale,
            variance_epsilon = 1e-12,
            name = name,
        )

        return out
