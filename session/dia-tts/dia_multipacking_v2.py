#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling
# task. Pointers for this are left as comments.

import torch

torch._dynamo.config.optimize_ddp=False

import numpy as np
torch.serialization.add_safe_globals(
    [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
)
import torch.nn.init as init
import torch.nn.functional as F
import math
import torch.nn as nn
import torchaudio
import logging
import os
import sys
import warnings
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
import datasets
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
from cut_cross_entropy import linear_cross_entropy
from huggingface_hub import hf_hub_download
from config import DiaConfig
from layers import DiaModel, KVCache
from model import Dia
from audio import build_delay_indices, apply_audio_delay
import dac
import torch
import torchaudio
from glob import glob
import json
import random


logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    merged_file: Optional[str] = field(
        default=None, metadata={
            "help": "merged file"})

def pad_attention_mask_4d(attention_mask, maxlen = 2048):
    maxlen_right = maxlen
    maxlen_bottom = maxlen
    attention_mask = [
        F.pad(
            attention_mask[i],
            (0, maxlen_right - attention_mask[i].shape[-2], 0, maxlen_bottom - attention_mask[i].shape[-1])) for i in range(
            len(attention_mask))]
    return torch.stack(attention_mask)

def pad_3d_2d(tensor, max_size, value):
    padded = [
        F.pad(tensor[i], (0, 0, 0, max_size - tensor[i].shape[0]), value = value) for i in range(len(tensor))
    ]
    return torch.stack(padded)

def pad_attention_mask(attention_mask, maxlen_right, maxlen_bottom):
    attention_mask = [
        F.pad(
            attention_mask[i],
            (0, maxlen_right - attention_mask[i].shape[1], 0, maxlen_bottom - attention_mask[i].shape[0])) for i in range(
            len(attention_mask))]
    return torch.stack(attention_mask)

def block_diagonal_concat(*masks, dtype=torch.bool):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    return combined_mask

def block_diagonal_concat_cross(*masks, dtype=torch.bool):
    total_rows = sum(mask.size(0) for mask in masks)
    total_cols = sum(mask.size(1) for mask in masks)
    
    combined_mask = torch.zeros((total_rows, total_cols), dtype=dtype)
    
    current_row, current_col = 0, 0

    for mask in masks:
        rows, cols = mask.size()
        combined_mask[current_row:current_row + rows, current_col:current_col + cols] = mask
        current_row += rows
        current_col += cols
        
    return combined_mask

def new_path(f):
    return f.replace('_processed/', '_processed_trim_dac/').replace('.mp3', '.dac')

class TTS(nn.Module):
    def __init__(self):
        super(TTS, self).__init__()

        dia_cfg = DiaConfig.load('config.json')
        ckpt_file = hf_hub_download('nari-labs/Dia-1.6B', filename="dia-v0_1.pth")
        model = DiaModel(dia_cfg)
        model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
        self.model = model
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.decoder.use_gradient_checkpointing = True
        
    def forward(
        self, 
        x_ids,
        src_positions,
        enc_self_attn_mask,
        tgt,
        actual_tgt,
        tgt_pos,
        dec_self_attn_mask,
        dec_cross_attn_mask,
        **kwargs,
    ):
        encoder_out = self.model.encoder(
            x_ids=x_ids,
            src_positions=src_positions,
            attn_mask=enc_self_attn_mask,
        )
        B, T, C = tgt.shape
        device = tgt.device

        self_attention_cache = [
            KVCache(
                batch_size=B,
                num_heads=self.model.decoder.layers[i].self_attention.num_query_heads,
                max_len=T,
                head_dim=self.model.decoder.layers[i].self_attention.head_dim,
                device=device,
            )
            for i in range(self.model.decoder.num_layers)
        ]

        cross_attention_cache = self.model.decoder.precompute_cross_attention_kv(
            max_len=encoder_out.shape[1],
            encoder_out=encoder_out,
            src_positions=src_positions,
        )
        logits = self.model.decoder(
            tgt_ids_BxTxC=tgt,
            encoder_out=encoder_out,
            tgt_positions=tgt_pos,
            src_positions=src_positions,
            self_attn_mask=dec_self_attn_mask,
            cross_attn_mask=dec_cross_attn_mask,
            self_attention_cache=self_attention_cache,
            cross_attention_cache=cross_attention_cache,
            deterministic = False,
        )
        logits = logits[:, :-1]
        tgt = actual_tgt[:, 1:]
        loss = 0
        codebook_size = self.model.decoder.logits_dense.weight.shape[1]
        for i in range(codebook_size):
            ci_loss = linear_cross_entropy(
                logits.flatten(0, -2).to(torch.bfloat16), 
                self.model.decoder.logits_dense.weight[:,i].T.to(torch.bfloat16),
                tgt[:,:,i].reshape(-1),
            )
            loss += ci_loss / codebook_size
        return {'loss': loss}

def main():

    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model = TTS()
    codebook_size = model.model.decoder.logits_dense.weight.shape[1]
    config = model.model.config
    pad_tok = config.data.text_pad_value
    pad_val = config.data.audio_pad_value
    bos_val = config.data.audio_bos_value
    eos_val = config.data.audio_eos_value

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(
            self, 
            dataset,
            merged_file = None,
        ):
            self.dataset = load_dataset(dataset)['train']
            self.length = len(self.dataset)
            self.merged = None

            if merged_file is not None:
                with open(merged_file) as fopen:
                    self.merged = json.load(fopen)
                self.length = len(self.merged)

        def __getitem__(self, idx):
            if self.merged is None:
                indices = [self.dataset[idx]]
            else:
                indices = self.merged[idx]
            try:
                texts, encodings = [], []
                for idx in indices:
                    data = self.dataset[idx]
                    reference_audio = data['reference_audio'] 
                    reference_text = data['reference_text']
                    target_audio = data['target_audio']
                    target_text = data['target_text']
                    text = f'[S1] {reference_text}[S1] {target_text}'
                    files = [reference_audio, target_audio]
                    encodeds = []
                    for f in files:
                        new_f = new_path(f)
                        with open(new_f) as fopen:
                            d = json.load(fopen)
                        d = torch.tensor(d)
                        if d.shape[1] != codebook_size:
                            d = d.T
                        encodeds.append(d)

                    encoding = torch.concat(encodeds, dim = 0)
                    texts.append(text)
                    encodings.append(encoding)

                text_ids = []
                enc_self_attn_mask = []
                enc_lens = []
                src_pos = []

                for txt in texts:
                    b_full = txt.encode('utf-8')
                    arr = list(b_full)

                    text_ids.append(torch.tensor(arr, dtype=torch.long))
                    l = len(arr)
                    enc_lens.append(l)
                    enc_self_attn_mask.append(torch.ones(l, l).bool())
                    src_pos.append(torch.arange(l))

                enc_self_attn_mask = block_diagonal_concat(*enc_self_attn_mask)

                tgts = []
                tgt_lens = []
                dec_self_attn_mask = []
                tgt_pos = []
                bos_tokens = torch.full([1, codebook_size], bos_val)
                for encoding in encodings:
                    encoding = torch.concat([bos_tokens, encoding])
                    ori_labels = encoding.T.tolist()
                    delayed_ = []
                    skip = 0
                    for i in range(len(ori_labels)):
                        delayed_.append([bos_val] * skip + ori_labels[i] + [eos_val])
                        if i == 0:
                            skip += 8
                        else:
                            skip += 1
                    maxlen = max([len(delayed_[i]) for i in range(len(ori_labels))])
                    for i in range(len(ori_labels)):
                        delayed_[i] = delayed_[i] + [eos_val] * (maxlen - len(delayed_[i]))
                    delayed_ = torch.tensor(delayed_).T
                    L = len(delayed_)
                    tgts.append(delayed_)
                    tgt_lens.append(L)
                    dec_self_attn_mask.append(torch.tril(torch.ones((L, L), dtype=torch.bool)))
                    tgt_pos.append(torch.arange(L))

                dec_self_attn_mask = block_diagonal_concat(*dec_self_attn_mask)

                dec_cross_attn_mask = []
                for i in range(len(tgt_lens)):
                    dec_cross_attn_mask.append(torch.ones(tgt_lens[i], enc_lens[i]).bool())

                dec_cross_attn_mask = block_diagonal_concat_cross(*dec_cross_attn_mask)

                text_ids = torch.concat(text_ids)
                src_pos = torch.concat(src_pos)
                tgt_pos = torch.concat(tgt_pos)
                tgt = torch.concat(tgts)
                return {
                    'text_ids': text_ids,
                    'src_pos': src_pos,
                    'enc_self_attn_mask': enc_self_attn_mask,
                    'tgt': tgt,
                    'tgt_pos': tgt_pos,
                    'dec_self_attn_mask': dec_self_attn_mask,
                    'dec_cross_attn_mask': dec_cross_attn_mask
                }
            except Exception as e:
                print(e)

        def __len__(self):
            return self.length
        
    def collator(batch):
        batch = [b for b in batch if b is not None]
        text_ids = [b['text_ids'] for b in batch]
        src_pos = [b['src_pos'] for b in batch]
        enc_self_attn_mask = [b['enc_self_attn_mask'] for b in batch]
        tgt = [b['tgt'] for b in batch]
        tgt_pos = [b['tgt_pos'] for b in batch]
        dec_self_attn_mask = [b['dec_self_attn_mask'] for b in batch]
        dec_cross_attn_mask = [b['dec_cross_attn_mask'] for b in batch]
        
        text_ids = pad_sequence(text_ids, batch_first = True, padding_value = pad_tok)
        src_pos = pad_sequence(src_pos, batch_first = True, padding_value = 0)
        
        maxlen_right = max([enc_self_attn_mask[i].shape[1] for i in range(len(enc_self_attn_mask))])
        maxlen_bottom = max([enc_self_attn_mask[i].shape[0] for i in range(len(enc_self_attn_mask))])
        enc_self_attn_mask = pad_attention_mask(enc_self_attn_mask, maxlen_right, maxlen_bottom)
        enc_self_attn_mask = enc_self_attn_mask[:, None]
        
        maxlen_tgt = max([tgt[i].shape[0] for i in range(len(tgt))])
        tgt = pad_3d_2d(tgt, maxlen_tgt, pad_val)
        actual_tgt = tgt.clone()
        actual_tgt[actual_tgt == pad_val] = -100
        tgt_pos = pad_sequence(tgt_pos, batch_first = True, padding_value = 0)
        
        maxlen_right = max([dec_self_attn_mask[i].shape[1] for i in range(len(dec_self_attn_mask))])
        maxlen_bottom = max([dec_self_attn_mask[i].shape[0] for i in range(len(dec_self_attn_mask))])
        dec_self_attn_mask = pad_attention_mask(dec_self_attn_mask, maxlen_right, maxlen_bottom)
        dec_self_attn_mask = dec_self_attn_mask[:, None]
        
        maxlen_right = max([dec_cross_attn_mask[i].shape[1] for i in range(len(dec_cross_attn_mask))])
        maxlen_bottom = max([dec_cross_attn_mask[i].shape[0] for i in range(len(dec_cross_attn_mask))])
        dec_cross_attn_mask = pad_attention_mask(dec_cross_attn_mask, maxlen_right, maxlen_bottom)
        dec_cross_attn_mask = dec_cross_attn_mask[:, None]
        
        return {
            'x_ids': text_ids,
            'src_positions': src_pos,
            'enc_self_attn_mask': enc_self_attn_mask,
            'tgt': tgt,
            'actual_tgt': actual_tgt,
            'tgt_pos': tgt_pos,
            'dec_self_attn_mask': dec_self_attn_mask,
            'dec_cross_attn_mask': dec_cross_attn_mask,
        }

    dataset = DatasetFixed(
        dataset=data_args.train_file, 
        merged_file=data_args.merged_file,
    )
    print('dataset', len(dataset), dataset[0])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()