#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
Training the Whisper model for sequence to sequence speech recognition via teacher-student distillation.
"""
# You can also adapt this script for your own distillation tasks. Pointers
# for this are left as comments.

import logging
import os
import re
import shutil
import sys
import time
import json
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from torch.utils.data import DataLoader, Dataset
from datasets import Audio
from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    Trainer,
)
from transformers import WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder, 
    WhisperDecoder, 
    WhisperModel,
    shift_tokens_right,
)
from transformers import AutoModel, AutoFeatureExtractor
from transformers import AutoConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.utils import check_min_version
from streaming import LocalDataset
from cut_cross_entropy import linear_cross_entropy
import wandb

logger = get_logger(__name__)

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: str = field(
        default=None, metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
            "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."}, )
    max_label_length: int = field(
        default=384,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    feature_extractor: Any
    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features):

        features = [f for f in features if f is not None]

        audios = [feature['input_features'] for feature in features]
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        batch = self.feature_extractor(audios,return_tensors='pt')
        batch['attention_mask'] = batch.pop('input_features_mask')

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        labels = labels.masked_fill(labels_mask.ne(1), -100)

        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch

def mse_loss_with_mask(input, target, mask):
    loss = torch.nn.functional.mse_loss(input, target, reduction='none')
    loss = loss.mean(dim=-1)
    loss = loss * mask
    return loss.sum() / mask.sum()

class GemmaWhisper(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        encoder_config = AutoConfig.from_pretrained(
            'mesolitica/gemma-3n-e4b-it-audio-encoder', trust_remote_code = True)
        self.encoder = AutoModel.from_config(encoder_config, trust_remote_code = True)
        self.decoder = WhisperDecoder(config)
        
        self.projection = nn.Linear(
            self.encoder.config.text_config.hidden_size, self.decoder.config.d_model)
        
        self.post_projection = nn.Linear(
            self.decoder.config.d_model, self.decoder.config.d_model)
        self.post_layer_norm = nn.LayerNorm(self.decoder.config.d_model)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value
    
    def init_quantize_layer(self, centroid_path = None):
        self.quantize_vocab_size = getattr(self.config, 'quantize_vocab_size', 32768)
        self.quantize_ema_decay = getattr(self.config, 'quantize_ema_decay', 0.99)
        self.quantize_loss_scale = getattr(self.config, 'quantize_loss_scale', 10.0)
        self.quantize_commit_coefficient = getattr(self.config, 'quantize_commit_coefficient', 0.25)
        self.codebook = nn.Embedding(self.quantize_vocab_size, self.decoder.config.d_model)
        if centroid_path is not None:
            init_codes = np.load(centroid_path)
            self.codebook.weight.data.copy_(torch.from_numpy(init_codes))
            print(f'loaded codebook weight from {centroid_path}')
        self.codebook.weight.requires_grad = False

        self.register_buffer("ema_count", torch.ones(self.quantize_vocab_size, dtype=torch.float))
        self.register_buffer("ema_weight", self.codebook.weight.data.clone().float())
        self.quantize_ema_count = 1

        self.quantize_update_interval = getattr(self.config, 'quantize_update_interval', 50)
        self.quantize_restart_interval = getattr(self.config, 'quantize_restart_interval', 500)

        self.register_buffer("total_code_usage", torch.zeros(self.quantize_vocab_size))

    def apply_vq(self, hidden_states, attention_mask):
        batch_size, seq_len, dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, dim)
        distances = (
            flat_hidden.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(flat_hidden, self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(1)
        )

        indices = torch.argmin(distances, dim=1)
        print(indices)
        quantized = self.codebook(indices).view(batch_size, seq_len, dim)
        if self.training:
            encodings = F.one_hot(indices, self.quantize_vocab_size).float()
            encodings = encodings * attention_mask.reshape(-1, 1)
            n = torch.sum(encodings, dim=0)
            if is_dist_initialized():
                torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)

            p = n / n.sum()
            self.quantize_perplexity = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()
            self.num_active_codes = (n > 0).sum().item()
            self.total_code_usage[indices] = 1.0

            hidden_flat = flat_hidden.detach()
            dw = torch.matmul(encodings.t(), hidden_flat)
            if is_dist_initialized():
                torch.distributed.all_reduce(dw, op=torch.distributed.ReduceOp.SUM)

            self.ema_count = self.ema_count * self.quantize_ema_decay + (
                1 - self.quantize_ema_decay) * n
            total_count = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + 1e-5) / (
                total_count + self.quantize_vocab_size * 1e-5) * total_count
            self.ema_weight = self.ema_weight * self.quantize_ema_decay + (
                1 - self.quantize_ema_decay) * dw
            if self.quantize_ema_count % self.quantize_update_interval == 0:
                self.codebook.weight.data = self.ema_weight / self.ema_count.unsqueeze(1)
            self.quantize_loss = self.quantize_loss_scale * self.quantize_commit_coefficient * mse_loss_with_mask(
                                hidden_states, quantized.detach(), attention_mask)
            
            self._maybe_restart_codes(hidden_flat, attention_mask)
            self.quantize_ema_count += 1

            hidden_states = hidden_states + (quantized - hidden_states).detach()
        else:
            hidden_states = quantized
        
        return hidden_states, indices
    
    def _maybe_restart_codes(self, hidden_flat, attention_mask):
        if self.quantize_restart_interval is None:
            return

        if self.quantize_ema_count % self.quantize_restart_interval != 0:
            return
        
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        segment_vocab_size = self.quantize_vocab_size // world_size
        start_idx = segment_vocab_size * rank
        ema_count_segment = self.ema_count[start_idx: start_idx + segment_vocab_size]
        threshold = self.quantize_ema_decay ** self.quantize_restart_interval
        update_indices = (ema_count_segment < threshold).nonzero()[:, 0] + start_idx
        num_update = update_indices.shape[0]

        print('num_update', num_update)

        if num_update > 0:
            mask_flat = attention_mask.reshape(-1) > 0
            hidden_selected = hidden_flat[mask_flat]
            chosen_indices = (
                torch.randperm(len(hidden_selected), device=hidden_selected.device)[:num_update]
                if num_update <= len(hidden_selected)
                else torch.randint(0, len(hidden_selected), (num_update,), device=hidden_selected.device)
            )
            hidden_update = hidden_selected[chosen_indices]

            num_update = torch.as_tensor([num_update], dtype=torch.long, device=hidden_flat.device)
            num_update_list = [torch.as_tensor([0], dtype=torch.long, device=hidden_flat.device)
                               for _ in range(world_size)]
            torch.distributed.all_gather(num_update_list, num_update)

            update_indices_list = [
                torch.zeros(num.item(), dtype=torch.long, device=hidden_flat.device)
                for num in num_update_list]
            torch.distributed.all_gather(update_indices_list, update_indices)
            update_indices = torch.cat(update_indices_list)

            hidden_update_list = [
                torch.zeros(num.item(), hidden_flat.shape[-1], dtype=hidden_update.dtype,
                            device=hidden_flat.device) for num in num_update_list]
            torch.distributed.all_gather(hidden_update_list, hidden_update)
            hidden_update = torch.cat(hidden_update_list)

            self.codebook.weight.data[update_indices] = hidden_update.to(self.codebook.weight.data.dtype)
            self.ema_count[update_indices] = 1
            self.ema_weight[update_indices] = hidden_update.to(self.ema_weight.dtype)

            if rank == 0:
                print(f"[VQ] Restarted {len(update_indices)} dead codes.")
        
    def forward(
        self,
        input_features = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        decoder_inputs_embeds = None,
        decoder_position_ids = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features=input_features,
                input_features_mask=attention_mask,
            )
        
        hidden_states = self.projection(encoder_outputs[0])
        attention_mask = ~encoder_outputs[1]
        
        hidden_states, indices = self.apply_vq(hidden_states, attention_mask)
        self.indices = indices

        hidden_states = self.post_layer_norm(self.post_projection(hidden_states))
            
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
    
class GemmaWhisperForConditionalGeneration(WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GemmaWhisper(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions

        self.post_init()
    
    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings
    
    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()
    
    def forward(
        self,
        input_features = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        decoder_inputs_embeds = None,
        decoder_position_ids = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])
        
        loss = None
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class Model(GemmaWhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, input_features, attention_mask, decoder_input_ids, labels = None, **kwargs):
        super_out = self.model.forward(
            input_features=input_features, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            auto_shift_loss = linear_cross_entropy(
                embeddings.to(torch.bfloat16), 
                self.proj_out.weight.to(torch.bfloat16), 
                labels, 
                shift=False,
                impl="cce_kahan_full_c"
            )
            self.ce_loss = auto_shift_loss
            return {'loss': auto_shift_loss + self.model.quantize_loss}
        return super_out

class Callback(TrainerCallback):
    def __init__(self, model, main_process):
        self.model = model
        self.main_process = main_process
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.main_process and wandb.run is None:
            wandb.init()

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.main_process:
            wandb.log({
                "quantize_loss": self.model.model.quantize_loss,
                "quantize_perplexity": self.model.model.quantize_perplexity,
                "num_active_codes": self.model.model.num_active_codes,
                "total_code_usage": self.model.model.total_code_usage.sum().item(),
                "ce_loss": self.model.ce_loss.item(),
            }, step=state.global_step)

def main():
    parser = HfArgumentParser(
        (ModelArguments,
         DataTrainingArguments,
         Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    tokenizer = WhisperTokenizerFast.from_pretrained(model_args.model_name_or_path)

    timestamps = [
        AddedToken(
            "<|%.2f|>" % (i * 0.02),
            lstrip=False,
            rstrip=False) for i in range(
            1500 + 1)]
    tokenizer.add_tokens(timestamps)

    model = Model.from_pretrained(model_args.model_name_or_path, attn_implementation='sdpa')
    model.model.init_quantize_layer('centroids-v2.npy')

    for name, param in model.model.encoder.named_parameters():
        param.requires_grad = False
    
    same = (model.proj_out.weight == model.model.decoder.embed_tokens.weight).float().mean().tolist()
    assert same >= 0.99, "projection is not tied"
    
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v3-turbo')
    sampling_rate = feature_extractor.sampling_rate

    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length)

    decoder_start_token_id = model.config.decoder_start_token_id  # <|startoftranscript|>
    decoder_prev_token_id = tokenizer.all_special_ids[-3]  # <|startofprev|>

    class Train(Dataset):
        def __init__(self, folder):
            if folder.endswith('.json'):
                with open(folder) as fopen:
                    self.data = json.load(fopen)
            else:
                self.data = LocalDataset(folder)
            self.audio = Audio(sampling_rate=16000)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            try:
                audio = self.audio.decode_example(
                    self.audio.encode_example(
                        self.data[item]['audio_filename']))['array']
                input_str = '<|startoftranscript|>' + self.data[item]['text'] + tokenizer.eos_token

                token_ids = tokenizer(input_str, add_special_tokens=False).input_ids
                if len(token_ids) > max_label_length:
                    return None
                
                d = {
                    'input_features': audio,
                    'input_length': [len(audio)],
                    'labels': token_ids,
                }
                return d
            except Exception as e:
                print(e)

                return None

    train_dataset = Train(data_args.train_dataset_name)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=feature_extractor,
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        decoder_prev_token_id=decoder_prev_token_id,
        input_padding="max_length",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[Callback(model, is_main_process(training_args.local_rank))],
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
