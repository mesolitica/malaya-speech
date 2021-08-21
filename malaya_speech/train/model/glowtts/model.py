import tensorflow as tf
from . import alignment, attention, common, modules
from ..fastspeech.model import TFFastSpeechEmbeddings, TFFastSpeechEncoder, TFTacotronPostnet
from ..fastspeech2.model import FastSpeechVariantPredictor


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, config_glowtts, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.embeddings = TFFastSpeechEmbeddings(config, name='embeddings')
        self.encoder = TFFastSpeechEncoder(
            config.encoder_self_attention_params, name='encoder'
        )
        self.proj_w = FastSpeechVariantPredictor(
            config, dtype=tf.float32, name='duration_predictor'
        )
        hidden_size = config.encoder_self_attention_params.hidden_size
        self.prenet = config_glowtts.prenet
        if self.prenet:
            self.pre = modules.ConvReluNorm(hidden_size, hidden_size,
                                            kernel_size=5, n_layers=3, p_dropout=0.5)

        self.proj_m = tf.keras.layers.Conv1D(hidden_size, 1, padding='same')
        self.mean_only = config_glowtts.mean_only
        if not self.mean_only:
            self.proj_s = tf.keras.layers.Conv1D(hidden_size, 1, padding='same')

    def call(
        self,
        input_ids,
        training=True,
        **kwargs,
    ):
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings(
            [input_ids, speaker_ids], training=training
        )
        attention_mask = tf.cast(attention_mask, tf.float32)
        attention_mask_expand = tf.expand_dims(attention_mask, -1)
        if self.prenet:
            embedding_output = self.pre(embedding_output, attention_mask_expand)
        x = self.encoder(
            [embedding_output, attention_mask], training=training
        )
        x = x[0]
        x_dp = tf.identity(x)
        x_m = self.proj_m(x) * attention_mask_expand
        if not self.mean_only:
            x_logs = self.proj_s(x) * attention_mask_expand
        else:
            x_logs = tf.zeros_like(x_m)
        logw = self.proj_w(
            [x_dp, speaker_ids, attention_mask]
        )
        return x_m, x_logs, logw, attention_mask_expand


class FlowSpecDecoder(tf.keras.layers.Layer):
    def __init__(self, config_glowtts, **kwargs):
        super(FlowSpecDecoder, self).__init__(name='FlowSpecDecoder', **kwargs)
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout = 0.,
        n_split = 4,
        n_sqz = 2,
        sigmoid_scale = False,
        gin_channels = 0
