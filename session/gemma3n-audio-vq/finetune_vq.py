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

# monkey patch pytorch validation
def _is_valid_woq_optimization_pattern():
    def fn(match):
        assert all(k in match.kwargs for k in ("x", "weight", "scales"))
        try:
            x = match.kwargs["x"].meta["val"]
            weight = match.kwargs["weight"].meta["val"]
            print(x.dtype, weight.dtype, x.device)
            scales = match.kwargs["scales"].meta["val"]
            
            return (
                # For now, we only support woq mm kernels
                # with x.type=bfloat16 and w.type=int8
                x.dtype == torch.bfloat16
                and weight.dtype == torch.int8
                and scales.dtype == torch.bfloat16
                # _weight_int8pack_mm kernel only supports cpu now
                # TODO: add cuda kernel support instead of calling mul+sum
                and x.device.type == "cpu"
                and x.device == weight.device
                and x.device == scales.device
            )
        except Exception as e:
            print(e, match)
            return False

    return fn

from torch._inductor.fx_passes import quantization
quantization._is_valid_woq_optimization_pattern = _is_valid_woq_optimization_pattern

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset

import transformers
import random
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import AutoProcessor, AutoConfig, Gemma3nAudioEncoder
from transformers import AutoFeatureExtractor, PreTrainedModel, AutoModel
from transformers import Gemma3nAudioEncoder, Gemma3nConfig
from transformers import AutoFeatureExtractor, PreTrainedModel
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nMultimodalEmbedder
from transformers.trainer_utils import get_last_checkpoint
import librosa
import json
import string
import re
import numpy as np
import pandas as pd

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

logger = logging.getLogger(__name__)

punct = set('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~')
digits = set(string.digits)

mapping = {
    '‘': '\'',
    '“': '"',
    '”': '"',
    '–': '-',
    '—': '-',
    '’': '\'',
    '\t': '',
    '\n': '',
    '…': ' ',
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_codes = 16384, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.codebook.weight.T)
            + torch.sum(self.codebook.weight**2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_codes, device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.codebook.weight)

        quantized = quantized.view(*input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.view(input_shape[:-1])

class Audio(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.audio_tower = Gemma3nAudioEncoder(config.audio_config)
        self.embed_audio = Gemma3nMultimodalEmbedder(config.audio_config, config.text_config)

class GemmaAudio(PreTrainedModel):
    config_class = Gemma3nConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = Audio(config)

    def forward(self, input_features, input_features_mask, **kwargs):
        output = self.model.audio_tower(
            input_features, ~input_features_mask,
        )
        project = self.model.embed_audio(inputs_embeds = output[0])
        return project, output[1]

class Model(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = GemmaAudio(config)
        out_features = self.encoder.model.embed_audio.embedding_projection.out_features
        self.vq = VectorQuantizer(out_features)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_features, nhead=8), num_layers=6
        )
        self.lm_head = nn.Linear(out_features, config.vocab_size)
        
    def forward(self, input_features, input_features_mask, labels = None, **kwargs):
        output = self.encoder(
            input_features = input_features,
            input_features_mask = input_features_mask,
        )
        quantized, vq_loss, tokens = self.vq(output[0])
        quantized = quantized.repeat_interleave(3, dim=1)
        mask = (~output[1]).repeat_interleave(3, dim=1)
        out_transformer = self.transformer(quantized)
        logits = self.lm_head(out_transformer)
        if labels is None:
            return logits, tokens, mask
        else:
            input_lengths = mask.sum(dim=-1)
            labels_lengths = (labels != 1).sum(-1)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1) 
            ctc_loss = torch.nn.functional.ctc_loss
            ctc_loss = ctc_loss(
                log_probs = log_probs,
                targets = labels,
                input_lengths = input_lengths,
                target_lengths = labels_lengths,
                zero_infinity = True,
            )
            print(ctc_loss, log_probs.shape, labels.shape)
            return {'loss': ctc_loss + vq_loss}

def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level
        # at info here to have that default.
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

    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    with open('cv-17-vocab.json') as fopen:
        vocab = json.load(fopen)
    
    vocab = ['BLANK', 'PAD', 'UNK'] + vocab
    vocab = {c: no for no, c in enumerate(vocab)}
    rev_vocab = {v: k for k, v in vocab.items()}
    pad_id = 1

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.vocab_size = len(vocab)
    model = Model(config)
    model.encoder = model.encoder.from_pretrained(model_args.model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local):
            self.dataset = pd.read_parquet(local)

        def __getitem__(self, idx):
            data = self.dataset.iloc[idx]
            try:
                y, sr = librosa.load(data['audio_filename'], sr = feature_extractor.sampling_rate)
                s = data['sentence']
                t = s.lower()
                for k, v in mapping.items():
                    t = t.replace(k, v)
                t = [c for c in t if c not in punct]
                t = re.sub(r'[ ]+', ' ', ''.join(t)).strip()
                label = [vocab[c] for c in t]

                return {'y': y, 'label': torch.tensor(label)}
            except Exception as e:
                print(e)
                return None

        def __len__(self):
            return len(self.dataset)

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset))

    def collator(batch):
        batch = [b for b in batch if b is not None]
        audio = [b['y'] for b in batch]
        labels = [b['label'] for b in batch]

        max_label_len = max([l.shape[0] for l in labels])
        padded_labels = torch.full((len(batch), max_label_len), fill_value=pad_id, dtype=torch.long)
        for i, l in enumerate(labels):
            padded_labels[i, :l.shape[0]] = l

        inputs = feature_extractor(audio, return_tensors = 'pt')
        return {'labels': padded_labels, **inputs}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

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
