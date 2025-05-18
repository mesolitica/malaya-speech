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
    calculated_speech_tokens: Optional[bool] = field(
        default=False, metadata={"help": "Use calculated speech tokens"})

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

    if data_args.calculated_speech_tokens:
        device = 'cpu'
        mimi = None
    else:
        device = 'cuda'
        dia_cfg = DiaConfig.load('config.json')
        dac_model = dac.DAC.load(dac.utils.download()).to(device)

    model = TTS()
    codebook_size = model.model.decoder.logits_dense.weight.shape[1]
    config = model.model.config
    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    max_audio = config.data.audio_length

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(
            self, 
            dataset,
            merged_file = None,
            calculated_speech_tokens = False,
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
                return {
                    'text': text,
                    'encoding': encoding,
                }
            except Exception as e:
                print(e)

        def __len__(self):
            return self.length
        
    def collator(batch):
        batch = [b for b in batch if b is not None]
        texts = [b['text'] for b in batch]
        
        text_ids = []
        for txt in texts:
            b_full = txt.encode('utf-8')
            bts = b_full[:max_text]
            arr = list(bts) + [pad_tok] * (max_text - len(bts))
            text_ids.append(torch.tensor(arr, dtype=torch.long))

        src = torch.stack(text_ids)
        src_pos = torch.arange(max_text).unsqueeze(0).expand(src.size(0), -1)
        src_pad = src.ne(pad_tok)
        enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

        encodings = [b['encoding'] for b in batch]
        seq_lens = [min(e.size(0), max_audio) for e in encodings]
        batch_max = max(seq_lens)
        padded = [pad(e, (0, 0, 0, batch_max - e.size(0))) if e.size(0) < batch_max else e[:batch_max]
              for e in encodings]
        codes = torch.stack(padded)
        B, T, C = codes.shape
        t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
        delayed = apply_audio_delay(
            codes,
            config.data.audio_pad_value,
            config.data.audio_bos_value,
            (t_idx, idxs)
        )
        delayed = delayed[:, :max_audio, :]

        max_tgt_len = max_audio + 2
        pad_val = config.data.audio_pad_value
        bos_val = config.data.audio_bos_value
        eos_val = config.data.audio_eos_value

        tgt = torch.full((B, max_tgt_len, C), pad_val, dtype=torch.long)
        tgt[:, 0, :] = bos_val
        for i, L in enumerate(seq_lens):
            tgt[i, 1:1 + L, :] = delayed[i, :L, :]
            tgt[i, 1 + L, :] = eos_val
        
        actual_tgt = tgt.clone()
        actual_tgt[actual_tgt == pad_val] = -100

        tgt_pos = torch.arange(max_tgt_len).unsqueeze(0).expand(B, -1)
        tgt_pad = tgt.ne(pad_val).any(-1)

        causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len), dtype=torch.bool))
        dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
        dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

        d = {
            'x_ids': src,
            'src_positions': src_pos,
            'enc_self_attn_mask': enc_self_attn_mask,
            'tgt': tgt,
            'actual_tgt': actual_tgt,
            'tgt_pos': tgt_pos,
            'dec_self_attn_mask': dec_self_attn_mask,
            'dec_cross_attn_mask': dec_cross_attn_mask,
        }
        return d

    dataset = DatasetFixed(
        dataset=data_args.train_file, 
        calculated_speech_tokens=data_args.calculated_speech_tokens,
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