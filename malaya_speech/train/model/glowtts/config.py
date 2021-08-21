from ..fastspeech.config import Config as FastSpeechConfig


class Config_GlowTTS():
    def __init__(self, dict):
        for k, v in dict.items():
            if type(v) == dict:
                v = Config(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class SelfAttentionParams:
    def __init__(
        self,
        n_speakers,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        attention_head_size,
        intermediate_size,
        intermediate_kernel_size,
        hidden_act,
        output_attentions,
        output_hidden_states,
        initializer_range,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        layer_norm_eps,
        max_position_embeddings,
    ):
        self.n_speakers = n_speakers
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.intermediate_size = intermediate_size
        self.intermediate_kernel_size = intermediate_kernel_size
        self.hidden_act = hidden_act
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings


class Config(FastSpeechConfig):
    """Initialize FastPitch Config."""

    def __init__(
        self,
        variant_prediction_num_conv_layers=2,
        variant_kernel_size=9,
        variant_dropout_rate=0.5,
        variant_predictor_filter=256,
        variant_predictor_kernel_size=3,
        variant_predictor_dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variant_prediction_num_conv_layers = (
            variant_prediction_num_conv_layers
        )
        self.variant_predictor_kernel_size = variant_predictor_kernel_size
        self.variant_predictor_dropout_rate = variant_predictor_dropout_rate
        self.variant_predictor_filter = variant_predictor_filter
