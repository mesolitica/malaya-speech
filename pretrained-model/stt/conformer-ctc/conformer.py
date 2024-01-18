from torchaudio.models import Conformer
from torchaudio.models.rnnt import _TimeReduction
from transformers import PretrainedConfig, PreTrainedModel
import torch
from torch import nn
from typing import List, Tuple, Optional


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

    def forward(self, input, lengths, labels=None):
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
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
