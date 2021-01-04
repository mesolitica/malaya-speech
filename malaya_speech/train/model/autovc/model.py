import tensorflow as tf


class LinearNorm(tf.keras.layers.Layer):
    def __init__(self, out_dim, bias = True, **kwargs):
        super(LinearNorm, self).__init__(name = 'LinearNorm', **kwargs)
        self.linear_layer = tf.keras.layers.Dense(out_dim, use_bias = bias)

    def call(self, x):
        return self.linear_layer(x)


class ConvNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size = 1,
        stride = 1,
        padding = 'SAME',
        dilation = 1,
        bias = True,
        **kwargs,
    ):
        super(ConvNorm, self).__init__(name = 'ConvNorm', **kwargs)
        self.conv = tf.keras.layers.Conv1D(
            out_channels,
            kernel_size = kernel_size,
            strides = stride,
            padding = padding,
            dilation_rate = dilation,
            use_bias = bias,
        )

    def call(self, x):
        return self.conv(x)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim_neck, freq, **kwargs):
        super(Encoder, self).__init__(name = 'Encoder', **kwargs)
        self.dim_neck = dim_neck
        self.freq = freq

        self.convolutions = []
        for i in range(3):
            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(512, kernel_size = 5, stride = 1, dilation = 1)
            )
            convolutions.add(tf.keras.layers.BatchNormalization())
            self.convolutions.append(convolutions)

        self.lstm = tf.keras.Sequential()
        for i in range(2):
            self.lstm.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(dim_neck, return_sequences = True)
                )
            )

    def call(self, x, c_org, training = True):
        c_org = tf.tile(tf.expand_dims(c_org, 1), (1, tf.shape(x)[1], 1))
        x = tf.concat([x, c_org], axis = -1)
        for c in self.convolutions:
            x = c(x, training = training)
            x = tf.nn.relu(x)
        outputs = self.lstm(x)
        out_forward = outputs[:, :, : self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck :]

        lens = tf.shape(outputs)[1]

        codes = tf.TensorArray(
            dtype = tf.float32,
            size = 0,
            dynamic_size = True,
            infer_shape = True,
        )
        init_state = (0, 0, codes)

        def condition(i, counter, codes):
            return i < lens

        def body(i, counter, codes):
            c = tf.concat(
                [out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]],
                axis = -1,
            )
            return i + self.freq, counter + 1, codes.write(counter, c)

        _, _, codes = tf.while_loop(condition, body, init_state)
        codes = codes.stack()

        return codes


class Decoder(tf.keras.layers.Layer):
    def __init__(self, dim_pre, **kwargs):
        super(Decoder, self).__init__(name = 'Decoder', **kwargs)
        self.lstm1 = tf.keras.layers.LSTM(dim_pre, return_sequences = True)
        self.convolutions = []
        for i in range(3):
            convolutions = tf.keras.Sequential()
            convolutions.add(
                ConvNorm(dim_pre, kernel_size = 5, stride = 1, dilation = 1)
            )
            convolutions.add(tf.keras.layers.BatchNormalization())
            self.convolutions.append(convolutions)

        self.lstm = tf.keras.Sequential()
        for i in range(2):
            self.lstm.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(1024, return_sequences = True)
                )
            )

        self.linear_projection = LinearNorm(80)

    def call(self, x, training = True):
        x = self.lstm1(x)
        for c in self.convolutions:
            x = c(x, training = training)
            x = tf.nn.relu(x)

        x = self.lstm(x)
        return self.linear_projection(x)


class Postnet(tf.keras.layers.Layer):
    def __init__(
        self,
        n_conv_postnet = 5,
        postnet_conv_filters = 512,
        postnet_conv_kernel_sizes = 5,
        postnet_dropout_rate = 0.1,
        num_mels = 80,
        **kwargs,
    ):
        super(Postnet, self).__init__(name = 'Postnet', **kwargs)
        self.conv_batch_norm = []
        for i in range(n_conv_postnet):
            conv = tf.keras.layers.Conv1D(
                filters = postnet_conv_filters
                if i < n_conv_postnet - 1
                else num_mels,
                kernel_size = postnet_conv_kernel_sizes,
                padding = 'same',
            )
            batch_norm = tf.keras.layers.BatchNormalization(axis = -1)
            self.conv_batch_norm.append((conv, batch_norm))
        self.dropout = tf.keras.layers.Dropout(rate = postnet_dropout_rate)
        self.activation = [tf.nn.tanh] * (n_conv_postnet - 1) + [tf.identity]

    def call(self, x, training = True):
        outputs = x
        for i, (conv, bn) in enumerate(self.conv_batch_norm):
            outputs = conv(outputs)
            outputs = bn(outputs)
            outputs = self.activation[i](outputs)
            outputs = self.dropout(outputs, training = training)
        return outputs


class Model(tf.keras.Model):
    def __init__(self, dim_neck, dim_pre, freq, **kwargs):
        super(Model, self).__init__(name = 'autovc', **kwargs)
        self.encoder = Encoder(dim_neck, freq)
        self.decoder = Decoder(dim_pre)
        self.postnet = Postnet()

    def call_second(self, x, c_org, training = True):
        return self.encoder(x, c_org, training = training)

    def call(self, x, c_org, c_trg, training = True, **kwargs):
        codes = self.encoder(x, c_org, training = training)  # [STACK, B, D]

        tmp = tf.TensorArray(
            dtype = tf.float32,
            size = 0,
            dynamic_size = True,
            infer_shape = True,
        )
        init_state = (0, tmp)
        stack_size = tf.shape(codes)[0]

        def condition(i, tmp):
            return i < stack_size

        def body(i, tmp):
            c = tf.expand_dims(codes[i], 1)
            c = tf.tile(
                c, (1, tf.cast(tf.shape(x)[1] / stack_size, tf.int32), 1)
            )
            return i + 1, tmp.write(i, c)

        _, tmp = tf.while_loop(condition, body, init_state)
        tmp = tmp.stack()  # [STACK, B, T, D]

        code_exp = tf.reshape(
            tmp, (tf.shape(x)[0], -1, self.encoder.dim_neck * 2)
        )
        c_trg = tf.tile(tf.expand_dims(c_trg, 1), (1, tf.shape(x)[1], 1))
        encoder_outputs = tf.concat([code_exp, c_trg], axis = -1)
        mel_before = self.decoder(encoder_outputs, training = training)
        mel_after = self.postnet(mel_before, training = training) + mel_before

        return encoder_outputs, mel_before, mel_after, codes
