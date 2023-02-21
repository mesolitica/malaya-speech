import logging
import math
from collections import namedtuple
from typing import List, Tuple, Optional

import sentencepiece as spm
import torch
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.models import emformer_rnnt_model, Hypothesis, RNNTBeamSearch
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
CUDA_VISIBLE_DEVICES=1 python3 emformer-base.py --batch=24 --precision=32
"""

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


def emformer_rnnt_base(num_symbols: int) -> RNNT:
    r"""Builds basic version of Emformer-based :class:`~torchaudio.models.RNNT`.
    Args:
        num_symbols (int): The size of target token lexicon.
    Returns:
        RNNT:
            Emformer RNN-T model.
    """
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
    filtered_hypo_tokens = [
        [token_index for token_index in h[tokens_idx][1:] if token_index not in post_process_remove_list] for h in hypos
    ]
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
        self.model = emformer_rnnt_base(num_symbols=1024)
        self.loss = torchaudio.transforms.RNNTLoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-9)

    def _step(self, batch, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
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
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device), 20)
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
    MAXLEN = 15
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
        FunctionalModule(partial(torch.nn.functional.pad, pad=(0, 4))),
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

    train_dataset = MalayaDataset('/home/husein/speech-bahasa/malay-asr-train-shuffled.json', sp_model)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=7, collate_fn=batch)

    val_dataset = MalayaDataset('/home/husein/ssd1/speech-bahasa/malay-asr-test.json', sp_model)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, collate_fn=batch)

    model_directory = f'emformer-base-{precision}'
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
