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
import numpy as np
import soundfile as sf
import random
import librosa
from scipy.signal import butter, lfilter, resample
from torch.utils.data import DataLoader, Dataset
from datasets import Audio
from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    Trainer,
)
from transformers import AutoFeatureExtractor
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

        with torch.no_grad():
            batch = self.processor(
                audios,
                return_tensors='pt',
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                return_attention_mask=True,
            )

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

class Model(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, input_features, attention_mask, decoder_input_ids, labels = None, **kwargs):
        super_out = self.model.forward(
            input_features=input_features, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            use_cache=False,
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

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def reduce_volume(data, db_reduction=-15):
    factor = 10 ** (db_reduction / 20.0)
    return data * factor

def add_noise(data, snr_db=15):
    rms_signal = np.sqrt(np.mean(data**2))
    rms_noise = rms_signal / (10**(snr_db/20))
    noise = np.random.normal(0, rms_noise, size=data.shape)
    return data + noise

def clip_distortion(data, threshold=0.3):
    return np.clip(data, -threshold, threshold)

def random_dropout(data, drop_prob=0.02, chunk_size=200):
    out = data.copy()
    n = len(data)
    for i in range(0, n, chunk_size):
        if np.random.rand() < drop_prob:
            out[i:i+chunk_size] = 0
    return out

def augment(data, sr):
    x = reduce_volume(data, -15)

    x = bandpass_filter(x, 300, 3400, sr)

    down_sr = random.choice([5000, 6000, 8000])
    num_samples = int(len(x) * down_sr / sr)
    x = resample(x, num_samples)

    x = add_noise(x, snr_db=random.randint(20, 50)) 
    x = clip_distortion(x, threshold=0.25)  
    x = random_dropout(x, drop_prob=0.03, chunk_size=400)

    up_sr = 16000
    num_samples_up = int(len(x) * up_sr / down_sr)
    x = resample(x, num_samples_up)
    return x

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

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation='sdpa',
    )

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
                if random.random() > 0.7:
                    audio = augment(audio, 16000)

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
