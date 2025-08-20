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
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
import librosa
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
from transformers.utils import check_min_version
from streaming import LocalDataset
from cut_cross_entropy import linear_cross_entropy

logger = get_logger(__name__)


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

class GemmaWhisper(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        encoder_config = AutoConfig.from_pretrained(
            'malaysia-ai/gemma-3n-e4b-it-audio-encoder', trust_remote_code = True)
        self.encoder = AutoModel.from_config(encoder_config, trust_remote_code = True)
        self.decoder = WhisperDecoder(config)
        
        self.projection = nn.Linear(
            self.encoder.config.text_config.hidden_size, self.decoder.config.d_model)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value
        
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
            
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=self.projection(encoder_outputs[0]),
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
                embeddings, 
                self.proj_out.weight, 
                labels, 
                shift=False,
                impl="cce_kahan_full_c"
            )
            return {'loss': auto_shift_loss}
        return super_out

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

    feature_extractor = AutoFeatureExtractor.from_pretrained('mesolitica/gemma-3n-e4b-it-audio-encoder')
    tokenizer = WhisperTokenizerFast.from_pretrained(model_args.model_name_or_path)

    timestamps = [
        AddedToken(
            "<|%.2f|>" % (i * 0.02),
            lstrip=False,
            rstrip=False) for i in range(
            1500 + 1)]
    tokenizer.add_tokens(timestamps)

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation='sdpa',
    )
    model.model.encoder = model.model.encoder.from_pretrained('mesolitica/gemma-3n-e4b-it-audio-encoder', trust_remote_code = True)
    for name, param in model.model.encoder.named_parameters():
        param.requires_grad = False
    
    same = (model.proj_out.weight == model.model.decoder.embed_tokens.weight).float().mean().tolist()
    assert same >= 0.99, "projection is not tied"

    processor = WhisperProcessor.from_pretrained(model_args.model_name_or_path)
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

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            try:
                audio = librosa.load(self.data[item]['audio_filename'], sr = 16000)[0]
                input_str = self.data[item]['text']

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
