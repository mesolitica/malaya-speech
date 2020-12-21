from ..fastspeech.config import Config as FastSpeechConfig


class Config(FastSpeechConfig):
    """Initialize FastSpeech2 Config."""

    def __init__(
        self,
        variant_prediction_num_conv_layers = 2,
        variant_kernel_size = 9,
        variant_dropout_rate = 0.5,
        variant_predictor_filter = 256,
        variant_predictor_kernel_size = 3,
        variant_predictor_dropout_rate = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variant_prediction_num_conv_layers = (
            variant_prediction_num_conv_layers
        )
        self.variant_predictor_kernel_size = variant_predictor_kernel_size
        self.variant_predictor_dropout_rate = variant_predictor_dropout_rate
        self.variant_predictor_filter = variant_predictor_filter
