"""
https://github.com/pytorch/audio/blob/main/examples/asr/librispeech_conformer_rnnt/transforms.py
"""

import torch
import math
import logging
import numpy as np
import json
from packaging import version
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

torchaudio_available = False
torchaudio_version = None

minimum_torchaudio_version = version.parse('0.13.1')

try:
    import torchaudio
    from torchaudio.models import emformer_rnnt_model, Hypothesis, RNNTBeamSearch
    from torchaudio.models import Conformer, RNNT
    from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber

    torchaudio_version = version.parse(torchaudio.__version__.split('+cu')[0])
    torchaudio_available = True
except Exception as e:
    logger.warning(f'torchaudio exception: {e}')
    logger.warning(
        '`torchaudio` is not available, `malaya_speech.utils.torch_featurization` is not able to use.')

    torchaudio = None
    StreamReader = None
    Hypothesis = None
    RNNTBeamSearch = None

try:
    from torchaudio.io import StreamReader
except Exception as e:
    logger.warning(f'torchaudio.io.StreamReader exception: {e}')
    logger.warning(
        '`torchaudio.io.StreamReader` is not available, `malaya_speech.streaming.torchaudio.stream` is not able to use.')
    StreamReader = None

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

if torchaudio_available:
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)
else:
    spectrogram_transform = None


class _ConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        time_reduction_stride: int,
        conformer_input_dim: int,
        conformer_ffn_dim: int,
        conformer_num_layers: int,
        conformer_num_heads: int,
        conformer_depthwise_conv_kernel_size: int,
        conformer_dropout: float,
    ) -> None:
        super().__init__()
        self.time_reduction = _TimeReduction(time_reduction_stride)
        self.input_linear = torch.nn.Linear(input_dim * time_reduction_stride, conformer_input_dim)
        self.conformer = Conformer(
            num_layers=conformer_num_layers,
            input_dim=conformer_input_dim,
            ffn_dim=conformer_ffn_dim,
            num_heads=conformer_num_heads,
            depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
            dropout=conformer_dropout,
            use_group_norm=True,
            convolution_first=True,
        )
        self.output_linear = torch.nn.Linear(conformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor,
                lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, lengths = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, lengths

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise RuntimeError("Conformer does not support streaming inference.")


def conformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_hidden_dim: int,
    lstm_layer_norm: int,
    lstm_layer_norm_epsilon: int,
    lstm_dropout: int,
    joiner_activation: str,
):
    r"""
    Builds Conformer-based recurrent neural network transducer (RNN-T) model.
    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        conformer_input_dim (int): dimension of Conformer input.
        conformer_ffn_dim (int): hidden layer dimension of each Conformer layer's feedforward network.
        conformer_num_layers (int): number of Conformer layers to instantiate.
        conformer_num_heads (int): number of attention heads in each Conformer layer.
        conformer_depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        conformer_dropout (float): Conformer dropout probability.
        num_symbols (int): cardinality of set of target tokens.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.
        joiner_activation (str): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
        Returns:
            RNNT:
                Conformer RNN-T model.
    """
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )
    predictor = _Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols, activation=joiner_activation)
    return RNNT(encoder, predictor, joiner)


def post_process_hypos(
    hypos: List[Hypothesis], sp_model
) -> List[Tuple[str, float, List[int], List[int]]]:
    tokens_idx = 0
    score_idx = 3
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    filtered_hypo_tokens = [[token_index for token_index in h[tokens_idx]
                             [1:] if token_index not in post_process_remove_list] for h in hypos]
    hypos_str = [sp_model.decode(s) for s in filtered_hypo_tokens]
    hypos_ids = [h[tokens_idx][1:] for h in hypos]
    hypos_score = [[math.exp(h[score_idx])] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))

    return nbest_batch


def validate_torchaudio():
    if not torchaudio_available:
        raise ModuleNotFoundError(
            'torchaudio not installed. Please install it by `pip install torchaudio` and try again.'
        )

    if torchaudio_version < minimum_torchaudio_version:
        raise ModuleNotFoundError(
            'torchaudio must minimum version 0.13.1. Please install it by `pip install torchaudio` and try again.'
        )


def conformer_rnnt_base():

    validate_torchaudio()

    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        time_reduction_stride=4,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation='tanh',
    )


def conformer_rnnt_tiny():

    validate_torchaudio()

    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        time_reduction_stride=4,
        conformer_input_dim=144,
        conformer_ffn_dim=576,
        conformer_num_layers=8,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation='tanh',
    )


def conformer_rnnt_medium():

    validate_torchaudio()

    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        time_reduction_stride=4,
        conformer_input_dim=384,
        conformer_ffn_dim=1536,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
    )


def conformer_rnnt_large():

    validate_torchaudio()

    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        time_reduction_stride=4,
        conformer_input_dim=512,
        conformer_ffn_dim=2048,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
    )


def emformer_rnnt_base(num_symbols: int = 1024):

    validate_torchaudio()

    return emformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        num_symbols=num_symbols,
        segment_length=16,
        right_context_length=4,
        time_reduction_input_dim=128,
        time_reduction_stride=4,
        transformer_num_heads=8,
        transformer_ffn_dim=1024,
        transformer_num_layers=16,
        transformer_dropout=0.1,
        transformer_activation="gelu",
        transformer_left_context_length=30,
        transformer_max_memory_size=0,
        transformer_weight_init_scale_strategy="depthwise",
        transformer_tanh_on_mem=True,
        symbol_embedding_dim=512,
        num_lstm_layers=2,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-3,
        lstm_dropout=0.3,
    )


def extract_labels(sp_model, samples):
    targets = [sp_model.encode(sample[2].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


def melspectrogram(x):
    if spectrogram_transform is None:
        raise ModuleNotFoundError(
            'torchaudio not installed. Please install it by `pip install torchaudio` and try again.'
        )
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    return spectrogram_transform(x).transpose(1, 0)


def piecewise_linear_log(x):
    x = x * GAIN
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob['mean'])
        self.invstddev = torch.tensor(blob['invstddev'])

    def forward(self, input):
        return (input - self.mean) * self.invstddev


class FeatureExtractor(torch.nn.Module):
    def __init__(self, global_stats_path, pad=False):
        super().__init__()

        self.global_stats = GlobalStatsNormalization(global_stats_path=global_stats_path)
        self.pad = pad

    def forward(self, input):
        mel = melspectrogram(input)
        mel = piecewise_linear_log(mel)
        mel = self.global_stats(mel)
        if self.pad:
            mel = torch.nn.functional.pad(mel, pad=(0, 0, 0, 4))
        return mel, torch.tensor([len(mel)])
