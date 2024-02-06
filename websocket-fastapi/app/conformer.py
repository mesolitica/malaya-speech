from torchaudio.models import Conformer
from torchaudio.models.rnnt import _TimeReduction
from transformers import PretrainedConfig, PreTrainedModel
import torch
import torchaudio
import math
import numpy as np
from torch import nn
from typing import List, Tuple, Optional

HF_CTC_VOCAB = [
    '',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    ' ',
    '?',
    '_'
]

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)


def piecewise_linear_log(x):
    x = x * GAIN
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


def melspectrogram(x):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    x = spectrogram_transform(x).transpose(1, 0)
    return piecewise_linear_log(x)


class ConformerConfig(PretrainedConfig):
    model_type = 'conformer'


class ConformerEncoder(PreTrainedModel):
    config_class = ConformerConfig

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__(config)
        self.time_reduction = _TimeReduction(config.time_reduction_stride)
        self.input_linear = torch.nn.Linear(
            config.input_dim * config.time_reduction_stride,
            config.conformer_input_dim)
        self.conformer = Conformer(
            num_layers=config.conformer_num_layers,
            input_dim=config.conformer_input_dim,
            ffn_dim=config.conformer_ffn_dim,
            num_heads=config.conformer_num_heads,
            depthwise_conv_kernel_size=config.conformer_depthwise_conv_kernel_size,
            dropout=config.conformer_dropout,
            use_group_norm=True,
            convolution_first=True,
        )
        self.output_linear = torch.nn.Linear(config.conformer_input_dim, config.output_dim)

    def forward(self, inputs, lengths, labels=None):
        time_reduction_out, time_reduction_lengths = self.time_reduction(inputs, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, input_lengths = self.conformer(input_linear_out, time_reduction_lengths)
        logits = self.output_linear(x)

        loss = None
        if labels is not None:
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            log_probs = nn.functional.log_softmax(
                logits,
                dim=-1,
                dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        output = (logits, input_lengths)
        return ((loss,) + output) if loss is not None else output
