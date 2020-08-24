import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from char_encoder import VOCAB_SIZE


def pad_second_dim(x, desired_size):
    padding = tf.tile(
        [[0]], tf.stack([tf.shape(x)[0], desired_size - tf.shape(x)[1]], 0)
    )
    return tf.concat([x, padding], 1)


class SpeechModel:
    def __init__(self, inputs):
        self.convolution_count = 0

        self.seq_lens = tf.count_nonzero(
            tf.reduce_sum(inputs, -1), 1, dtype = tf.int32
        )

        outputs = tf.layers.conv1d(
            inputs,
            filters = 250,
            kernel_size = 48,
            stride = 2,
            padding = 'SAME',
            activation = tf.nn.relu,
        )
        for layer_idx in range(7):
            outputs = tf.layers.conv1d(
                outputs,
                filters = 250,
                kernel_size = 7,
                stride = 1,
                padding = 'SAME',
                activation = tf.nn.relu,
            )

        outputs = tf.layers.conv1d(
            outputs,
            filters = 250 * 8,
            kernel_size = 32,
            stride = 1,
            padding = 'SAME',
            activation = tf.nn.relu,
        )

        outputs = tf.layers.conv1d(
            outputs,
            filters = 250 * 8,
            kernel_size = 1,
            stride = 1,
            padding = 'SAME',
            activation = tf.nn.relu,
        )

        outputs = tf.layers.conv1d(
            outputs,
            filters = VOCAB_SIZE,
            kernel_size = 1,
            stride = 1,
            padding = 'SAME',
            activation = None,
        )

        self.logits = outputs

    def calculate_loss_accuracy(self, label, Y):
        Y_seq_len = tf.count_nonzero(label, 1, dtype = tf.int32)
        filled = tf.fill(tf.shape(self.seq_lens), tf.reduce_max(Y_seq_len))
        seq_lens = tf.where(
            seq_lens < tf.reduce_max(Y_seq_len), filled, seq_lens
        )

        self.cost = tf.reduce_mean(tf.nn.ctc_loss(Y, self.logits, seq_lens))

        decoded, log_prob = tf.nn.ctc_greedy_decoder(self.logits, seq_lens)
        decoded = tf.to_int32(decoded[0])
        preds = tf.sparse.to_dense(decoded)
        preds = preds[:, : tf.reduce_max(Y_seq_len)]
        masks = tf.sequence_mask(
            Y_seq_len, tf.reduce_max(Y_seq_len), dtype = tf.float32
        )

        preds = pad_second_dim(preds, tf.reduce_max(Y_seq_len))
        y_t = tf.cast(preds, tf.int32)
        prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.label, masks)
        correct_pred = tf.equal(prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
