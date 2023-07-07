import logging
import math
from collections import namedtuple
from typing import List, Tuple, Optional

import sentencepiece as spm
import torch
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.models import Hypothesis, RNNTBeamSearch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import malaya_speech
from malaya_speech.utils import torch_featurization
import numpy as np
import json
from datasets import Audio
from functools import partial
import random
from transformers import get_linear_schedule_with_warmup
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', help='batch size')
parser.add_argument('-c', '--checkpoint', help='checkpoint')
parser.add_argument('-g', '--gradient_clipping', help='gradient clipping')
parser.add_argument('-p', '--precision', help='precision')
parser.add_argument('-a', '--accumulate', help='accumulate')
parser.add_argument('-s', '--save', help='save steps')
args = parser.parse_args()
batch_size = int(args.batch)
ckpt_path = args.checkpoint
precision = int(args.precision or 16)
accumulate = int(args.accumulate or 1)
save_steps = int(args.save or 5000)

"""
CUDA_VISIBLE_DEVICES=0 python3 large-noisy.py --batch=10 --precision=32
"""

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


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
) -> RNNT:
    r"""Builds Conformer-based recurrent neural network transducer (RNN-T) model.
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


def conformer_rnnt_base() -> RNNT:
    r"""Builds basic version of Conformer RNN-T model.
    Returns:
        RNNT:
            Conformer RNN-T model.
    """
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


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Learning rate scheduler that performs linear warmup and exponential annealing.
    Args:
        optimizer (torch.optim.Optimizer): optimizer to use.
        warmup_steps (int): number of scheduler steps for which to warm up learning rate.
        force_anneal_step (int): scheduler step at which annealing of learning rate begins.
        anneal_factor (float): factor to scale base learning rate by at each annealing step.
        last_epoch (int, optional): The index of last epoch. (Default: -1)
        verbose (bool, optional): If ``True``, prints a message to stdout for
            each update. (Default: ``False``)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        force_anneal_step: int,
        anneal_factor: float,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.force_anneal_step = force_anneal_step
        self.anneal_factor = anneal_factor
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.force_anneal_step:
            return [(min(1.0, self._step_count / self.warmup_steps))
                    * base_lr for base_lr in self.base_lrs]
        else:
            scaling_factor = self.anneal_factor ** (self._step_count - self.force_anneal_step)
            return [scaling_factor * base_lr for base_lr in self.base_lrs]


def post_process_hypos(
    hypos: List[Hypothesis], sp_model: spm.SentencePieceProcessor
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


class ConformerRNNTModule(LightningModule):
    def __init__(self, sp_model):
        super().__init__()

        self.sp_model = sp_model
        spm_vocab_size = self.sp_model.get_piece_size()
        self.blank_idx = spm_vocab_size

        # ``conformer_rnnt_base`` hardcodes a specific Conformer RNN-T configuration.
        # For greater customizability, please refer to ``conformer_rnnt_model``.
        self.model = conformer_rnnt_base()
        self.loss = torchaudio.transforms.RNNTLoss(reduction="mean")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=5e-5,
            betas=(
                0.9,
                0.98),
            eps=1e-9)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 40, 120, 0.96)

    def _step(self, batch, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty(
            [batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _ = self.model(
            batch.features,
            batch.feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=50000,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return (
            [self.optimizer],
            [scheduler],
        )

    def forward(self, batch: Batch):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        hypotheses = decoder(
            batch.features.to(
                self.device), batch.feature_lengths.to(
                self.device), 20)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

    def training_step(self, batch: Batch, batch_idx):
        """Custom training step.
        By default, DDP does the following on each train step:
        - For each GPU, compute loss and gradient on shard of training data.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / N, where N is the world
          size (total number of GPUs).
        - Update parameters on each GPU.
        Here, we do the following:
        - For k-th GPU, compute loss and scale it by (N / B_total), where B_total is
          the sum of batch sizes across all GPUs. Compute gradient from scaled loss.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / B_total.
        - Update parameters on each GPU.
        Doing so allows us to account for the variability in batch sizes that
        variable-length sequential data yield.
        """
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")


def check_nan(a):
    return np.isnan(np.sum(a))


def check_inf(a):
    return np.isinf(np.sum(a))


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class MalayaDataset(torch.utils.data.Dataset):

    SR = 16000
    MAXLEN = 16
    MINLEN = 0.1
    MINLEN_TEXT = 1
    AUGMENTATION = False

    def __init__(self, file, sp_model):
        with open(file) as fopen:
            self.data = json.load(fopen)
        self.sp_model = sp_model
        self.global_stats = torch_featurization.GlobalStatsNormalization('malay-stats.json')
        self.audio = Audio(sampling_rate=self.SR)

    def __getitem__(self, idx):
        x = self.data['X'][idx]
        y = self.data['Y'][idx]

        r = self.audio.decode_example(self.audio.encode_example(x))

        if (len(r['array']) / self.SR) > self.MAXLEN:
            return

        if (len(r['array']) / self.SR) < self.MINLEN:
            return

        if len(y) < self.MINLEN_TEXT:
            return

        mel = torch_featurization.melspectrogram(r['array'])
        mel = torch_featurization.piecewise_linear_log(mel)
        mel = self.global_stats(mel)

        y_ = sp_model.encode(y.lower())

        batch = {
            'feature': mel,
            'feature_len': len(mel),
            'target': y_,
            'target_len': len(y_)
        }

        return batch

    def __len__(self):
        return len(self.data['X'])


if __name__ == '__main__':

    sp_model = spm.SentencePieceProcessor(model_file='/home/husein/malaya-speech/malay-tts.model')

    train_data_pipeline = torch.nn.Sequential(
        FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        torchaudio.transforms.FrequencyMasking(20),
        torchaudio.transforms.FrequencyMasking(20),
        torchaudio.transforms.TimeMasking(80, p=0.2),
        torchaudio.transforms.TimeMasking(80, p=0.2),
        FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
    )

    def batch(batches):
        features, feature_lens, targets, target_lens = [], [], [], []
        for i in range(len(batches)):
            b = batches[i]
            if b is not None:
                features.append(b['feature'])
                feature_lens.append(b['feature_len'])
                targets.append(b['target'])
                target_lens.append(b['target_len'])

        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        features = train_data_pipeline(features)
        feature_lens = torch.tensor(feature_lens, dtype=torch.int32)

        target_lens = torch.tensor(target_lens).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)

        return Batch(features, feature_lens, targets, target_lens)

    train_dataset = MalayaDataset(
        '/home/husein/malaya-speech/malay-stt-train-augmented.json', sp_model)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=7,
        collate_fn=batch)

    val_dataset = MalayaDataset('/home/husein/ssd1/speech-bahasa/malay-asr-test.json', sp_model)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, collate_fn=batch)

    model_directory = f'conformer-large-v2-whisper-{precision}'
    model = ConformerRNNTModule(sp_model)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='step',
        mode='max',
        dirpath=model_directory,
        every_n_train_steps=save_steps,
        filename='model-{epoch:02d}-{step}',
    )

    num_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',
        devices=num_gpus,
        limit_val_batches=100,
        precision=precision,
        accumulate_grad_batches=accumulate,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
