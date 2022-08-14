import tensorflow.compat.v1 as tf
import math
from ..utils import shape_list


def index_put_constant(tensor, indices, value):
    tiled = tf.fill(tf.shape(tensor), value)
    return tf.where(indices, tiled, tensor)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = tf.keras.layers.Conv1D(channels, 1, padding='SAME')
        self.conv_k = tf.keras.layers.Conv1D(channels, 1, padding='SAME')
        self.conv_v = tf.keras.layers.Conv1D(channels, 1, padding='SAME')
        self.conv_o = tf.keras.layers.Conv1D(out_channels, 1, padding='SAME')
        self.drop = tf.keras.layers.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = tf.Variable(tf.random.normal(
                (n_heads_rel, window_size * 2 + 1, self.k_channels), dtype=tf.float32) * rel_stddev)
            self.emb_rel_v = tf.Variable(tf.random.normal(
                (n_heads_rel, window_size * 2 + 1, self.k_channels), dtype=tf.float32) * rel_stddev)

    def call(self, x, c, attn_mask=None, training=True):
        # x [B, T, D]
        # c [B, Tc, Dc]
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask, training=training)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None, training=True):
        b, t_s, d = shape_list(key)
        _, t_t, _ = shape_list(query)

        # [B, t_t, N, K]
        query = tf.reshape(query, (b, t_t, self.n_heads, self.k_channels))
        # [B, N, t_t, K]
        query = tf.transpose(query, [0, 2, 1, 3])

        # [B, t_s, N, K]
        key = tf.reshape(key, (b, t_s, self.n_heads, self.k_channels))
        # [B, N, t_s, K]
        key = tf.transpose(key, [0, 2, 1, 3])

        # [B, t_s, N, K]
        value = tf.reshape(value, (b, t_s, self.n_heads, self.k_channels))
        # [B, N, t_s, K]
        value = tf.transpose(value, [0, 2, 1, 3])

        # [B, N, t_t, K] * [B, N, K, t_s] = [B, H, t_t, t_s]
        scores = tf.matmul(query / math.sqrt(self.k_channels), tf.transpose(key, [0, 1, 3, 2]))

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            scores = scores + self._attention_bias_proximal(t_s)
        if mask is not None:
            # mask must shape [B, H, t_t, t_s]
            mask_ = tf.tile(mask, [1, self.n_heads, 1, 1])
            scores = index_put_constant(scores, tf.equal(mask_, 0), 1e-4)
        p_attn = tf.nn.softmax(scores, axis=-1)
        p_attn = self.drop(p_attn, training=training)
        # [B, H, t_t, t_s] * [B, N, t_s, K] = [B, H, t_t, K]
        output = tf.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), [b, t_t, d])
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = tf.matmul(x, tf.expand_dims(y, 0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        # [b, 1, 9, 96] -> [b, 1]
        ret = tf.matmul(x, tf.transpose(tf.expand_dims(y, 0), [0, 1, 3, 2]))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = tf.math.maximum(length - (self.window_size + 1), 0)
        slice_start_position = tf.math.maximum((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1

        padded_relative_embeddings = tf.cond(tf.greater(pad_length, 0),
                                             lambda: tf.pad(relative_embeddings, [
                                                            [0, 0], [pad_length, pad_length], [0, 0]]),
                                             lambda: relative_embeddings)
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = shape_list(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])
        x_flat = tf.reshape(x, [batch, heads, length * 2 * length])
        x_flat = tf.pad(x_flat, [[0, 0], [0, 0], [0, length-1]])
        x_final = tf.reshape(x_flat, [batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = shape_list(x)
        # padd along column
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, length-1]])
        x_flat = tf.reshape(x, [batch, heads, length**2 + length*(length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = tf.pad(x_flat, [[0, 0], [0, 0], [length, 0]])
        x_final = tf.reshape(x_flat, [batch, heads, length, 2*length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
        length: an integer scalar.
        Returns:
        a Tensor with shape [1, 1, length, length]
        """
        r = tf.range(length, dtype=tf.float32)
        diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
        return tf.expand_dims(tf.expand_dims(-tf.math.log1p(tf.abs(diff)), 0), 0)


class FFN(tf.keras.layers.Layer):
    def __init__(self, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, **kwargs):
        super(FFN, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv_1 = tf.keras.layers.Conv1D(filter_channels, kernel_size, padding='SAME')
        self.conv_2 = tf.keras.layers.Conv1D(out_channels, kernel_size, padding='SAME')
        self.drop = tf.keras.layers.Dropout(p_dropout)

    def call(self, x, x_mask, training=True):
        x = self.conv_1(x * x_mask)
        if self.activation == "gelu":
            x = x * tf.sigmoid(1.702 * x)
        else:
            x = tf.nn.relu(x)
        x = self.drop(x, training=training)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(tf.keras.layers.Layer):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = tf.keras.layers.Dropout(p_dropout)
        self.attn_layers = []
        self.norm_layers_1 = []
        self.ffn_layers = []
        self.norm_layers_2 = []

        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels,
                                    n_heads, p_dropout=p_dropout, window_size=window_size))
            self.norm_layers_1.append(tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5))
            self.ffn_layers.append(FFN(hidden_channels,
                                   filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5))

    def call(self, x, x_mask, training=True):
        x_mask_t = tf.transpose(x_mask, [0, 2, 1])
        attn_mask = tf.expand_dims(x_mask_t, 2) * tf.expand_dims(x_mask_t, -1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y, training=training)
            x = self.norm_layers_1[i](x + y, training=training)
            y = self.ffn_layers[i](x, x_mask, training=training)
            y = self.drop(y, training=training)
            x = self.norm_layers_2[i](x + y, training=training)
        x = x * x_mask
        return x
