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
import math
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import logging
import os
import sys
import warnings
from torch.nn.utils.rnn import pad_sequence
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
from moshi.models import loaders
from glob import glob
from models import Model
from generator import load_llama3_tokenizer
import json
import random

logger = logging.getLogger(__name__)

def pad_2d(tensor, max_size, value):
    padded = [
        F.pad(tensor[i], (0, max_size - tensor[i].shape[0]), value = value) for i in range(len(tensor))
    ]
    return torch.stack(padded)

def pad_3d_2d(tensor, max_size, value):
    padded = [
        F.pad(tensor[i], (0, 0, 0, max_size - tensor[i].shape[0]), value = value) for i in range(len(tensor))
    ]
    return torch.stack(padded)

def pad_3d_all(tensor, max_size, value):
    padded = [
        F.pad(tensor[i], (0, max_size - tensor[i].shape[0], 0, max_size - tensor[i].shape[1]), value = value) for i in range(len(tensor))
    ]
    return torch.stack(padded)

def pad_attention_mask(attention_mask, max_size = 4096, value = False):
    maxlen_right = max_size
    maxlen_bottom = max_size
    attention_mask = [
        F.pad(
            attention_mask[i],
            (0, maxlen_right - attention_mask[i].shape[-2], 0, maxlen_bottom - attention_mask[i].shape[-1]), value = value) for i in range(
            len(attention_mask))]
    return torch.stack(attention_mask)

def block_diagonal_concat_inverted(*masks, dtype=torch.bool):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size
        
    return combined_mask

def _tokenize_text_segment(text_tokenizer, text: str, speaker: int, device):
    frame_tokens = []
    frame_masks = []

    text_tokens = text_tokenizer.encode(f"[{speaker}]{text}")
    text_frame = torch.zeros(len(text_tokens), 33).long()
    text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
    text_frame[:, -1] = torch.tensor(text_tokens)
    text_frame_mask[:, -1] = True

    frame_tokens.append(text_frame.to(device))
    frame_masks.append(text_frame_mask.to(device))

    return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

def _token_audio(audio_tokenizer, audio, device):
    audio_tensor, sample_rate = torchaudio.load(audio)
    if audio_tensor.shape[0] != 1:
        audio_tensor = audio_tensor.mean(dim=0)
    audio_tensor = audio_tensor.squeeze(0)
    if sample_rate != audio_tokenizer.sample_rate:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=audio_tokenizer.sample_rate
        )
    audio_tensor = audio_tensor.to(device)
    audio_tokens = audio_tokenizer.encode(audio_tensor.unsqueeze(0).unsqueeze(0))[0]
    return audio_tokens


def _tokenize_audio(audio_tokens, device):
    frame_tokens = []
    frame_masks = []

    eos_frame = torch.zeros(audio_tokens.size(0), 1).to(device)
    audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

    audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(device)
    audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(device)
    audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
    audio_frame_mask[:, :-1] = True

    frame_tokens.append(audio_frame)
    frame_masks.append(audio_frame_mask)

    return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

def new_path(f):
    return f.replace('_processed/', '_processed_trim_moshi/').replace('.mp3', '.moshi')

def get_input_ids(
    reference_audio, 
    reference_text, 
    target_audio,
    target_text,
    mimi,
    text_tokenizer,
    device,
    calculated_speech_tokens = False,
):
    if calculated_speech_tokens:
        reference_f = new_path(reference_audio)
        if not os.path.exists(reference_f):
            reference_f = ''.join(c for c in reference_f if ord(c) < 128)
        with open(reference_f) as fopen:
            d = json.load(fopen)
        reference_audio_tokens = torch.tensor(d, device = device)

        target_f = new_path(target_audio)
        if not os.path.exists(target_f):
            target_f = ''.join(c for c in target_f if ord(c) < 128)
        with open(target_f) as fopen:
            d = json.load(fopen)
        target_audio_tokens = torch.tensor(d, device = device)
    else:
        reference_audio_tokens = _token_audio(mimi, reference_audio, device)
        target_audio_tokens = _token_audio(mimi, target_audio, device)

    reference_audio_tokens, reference_audio_masks = _tokenize_audio(reference_audio_tokens, device)
    target_audio_tokens, target_audio_masks = _tokenize_audio(target_audio_tokens, device)
        
    reference_text_tokens, reference_text_masks = _tokenize_text_segment(
        text_tokenizer, reference_text, 0, device)
    target_text_tokens, target_text_masks = _tokenize_text_segment(
        text_tokenizer, target_text, 0, device)

    segment_tokens = torch.cat(
        [reference_text_tokens, reference_audio_tokens, target_text_tokens, target_audio_tokens], dim=0)
    segment_tokens_mask = torch.cat(
        [reference_text_masks, reference_audio_masks, target_text_masks, target_audio_masks], dim=0)
    return segment_tokens, segment_tokens_mask

@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    merged_file: Optional[str] = field(
        default=None, metadata={
            "help": "multipacking merge file, if provide, the training will become multipacking"})
    calculated_speech_tokens: Optional[bool] = field(
        default=False, metadata={"help": "Use calculated speech tokens"})
    max_length: Optional[int] = field(
        default=0, metadata={"help": "max length"})

class TTS(nn.Module):
    def __init__(self):
        super(TTS, self).__init__()

        self.model = Model.from_pretrained("sesame/csm-1b")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
        
    def forward(self, input_ids, attention_mask, labels, extra_mask = None, input_pos = None, **kwargs):
        shifted_audio_tokens = input_ids[:, 1:, :-1]
        labels_shifted_audio_tokens = labels[:, 1:, :-1]
        dtype = next(self.model.parameters()).dtype
        embeds = self.model._embed_tokens(input_ids)
        masked_embeds = embeds * attention_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.model.backbone(h, mask = extra_mask, input_pos = input_pos).to(dtype)
        h = h[:,:-1]
        c0_logits = self.model.codebook0_head(h)
        ci_stacked = torch.stack(
            [
                self.model._embed_audio(i, shifted_audio_tokens[:, :, i])
                for i in range(self.model.config.audio_num_codebooks)
            ],
            dim=-2,
        )
        decoder_inputs = torch.concat([h.unsqueeze(2), ci_stacked], dim=-2)
        batch_size = decoder_inputs.shape[0]
        seq_len = decoder_inputs.shape[1]
        decoder_inputs = decoder_inputs.reshape(
            -1, self.model.config.audio_num_codebooks + 1, decoder_inputs.shape[-1]
        )
        decoder_h = self.model.decoder(self.model.projection(decoder_inputs)).to(dtype=dtype)
        decoder_h = decoder_h.reshape(
            batch_size, seq_len, self.model.config.audio_num_codebooks + 1, -1
        )[:, :, 1:-1, :] 
        shift_embeddings = h.flatten(0, -2)
        c0_loss = linear_cross_entropy(
            shift_embeddings.to(torch.bfloat16), 
            self.model.codebook0_head.weight.to(torch.bfloat16),
            labels_shifted_audio_tokens[:, :, 0].reshape(-1)
        )
        total_loss = c0_loss / self.model.config.audio_num_codebooks
        for index in range(1, self.model.config.audio_num_codebooks):
            shift_embeddings = decoder_h[:, :, index - 1, :].flatten(0, -2)
            ci_loss = linear_cross_entropy(
                shift_embeddings.to(torch.bfloat16), 
                self.model.audio_head[index - 1].T.to(torch.bfloat16),
                labels_shifted_audio_tokens[:, :, index].reshape(-1),
            )
            total_loss += ci_loss / self.model.config.audio_num_codebooks
        return {'loss': total_loss}

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

    text_tokenizer = load_llama3_tokenizer()
    if data_args.calculated_speech_tokens:
        device = 'cpu'
        mimi = None
    else:
        device = 'cuda'
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)

    model = TTS()

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(
            self, 
            dataset,
            merged_file = None,
            calculated_speech_tokens = False,
        ):
            self.dataset = load_dataset(dataset)['train']
            self.length = len(self.dataset)
            self.calculated_speech_tokens = calculated_speech_tokens
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
                tokens, masks, lengths, pos = [], [], [], []
                for i in indices:
                    data = self.dataset[i]
                    segment_tokens, segment_tokens_mask = get_input_ids(
                        data['reference_audio'], 
                        data['reference_text'],
                        data['target_audio'], 
                        data['target_text'],
                        mimi=mimi,
                        text_tokenizer=text_tokenizer,
                        device=device,
                        calculated_speech_tokens=self.calculated_speech_tokens,
                    )
                    tokens.append(segment_tokens)
                    masks.append(segment_tokens_mask)
                    m = segment_tokens.shape[0]
                    lengths.append(torch.tril(torch.ones(m, m, dtype=torch.bool)))
                    pos.append(torch.arange(0, m))

                extra_mask = block_diagonal_concat_inverted(*lengths)
                return {
                    'input_ids': torch.concat(tokens, dim = 0),
                    'attention_mask': torch.concat(masks, dim = 0),
                    'extra_mask': extra_mask,
                    'input_pos': torch.concat(pos, dim = 0)
                }
                
            except Exception as e:
                print(e)

        def __len__(self):
            return self.length
        
    def collator(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [b['input_ids'] for b in batch]
        attention_mask = [b['attention_mask'] for b in batch]
        extra_mask = [b['extra_mask'] for b in batch]
        input_pos = [b['input_pos'] for b in batch]
        labels = pad_sequence(input_ids, batch_first = True, padding_value = -100)
        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = 0)
        attention_mask = pad_sequence(attention_mask, batch_first = True, padding_value = False)
        input_pos = pad_sequence(input_pos, batch_first = True, padding_value = 0)
        max_size = max([extra_mask[k].shape[0] for k in range(len(extra_mask))])
        extra_mask = pad_attention_mask(extra_mask, max_size = max_size)
        if data_args.max_length > 0:
            input_ids = input_ids[:, :data_args.max_length]
            attention_mask = attention_mask[:, :data_args.max_length]
            labels = labels[:, :data_args.max_length]
            extra_mask = extra_mask[:, :data_args.max_length, :data_args.max_length]
            input_pos = input_pos[:, :data_args.max_length]

            input_ids = pad_3d_2d(input_ids, data_args.max_length, 0)
            attention_mask = pad_3d_2d(attention_mask, data_args.max_length, False)
            labels = pad_3d_2d(labels, data_args.max_length, -100)
            extra_mask = pad_3d_all(extra_mask, data_args.max_length, False)
            input_pos = pad_2d(input_pos, data_args.max_length, 0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'extra_mask': extra_mask,
            'input_pos': input_pos,
        }

    dataset = DatasetFixed(
        dataset=data_args.train_file, 
        merged_file=data_args.merged_file,
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