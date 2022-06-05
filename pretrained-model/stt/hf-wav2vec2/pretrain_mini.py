#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['WANDB_DISABLED'] = 'true'

import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except BaseException:
    pass


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import sys
import string
import torch
import numpy as np
import transformers
import requests
import datasets
import torch
from torch import nn
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from packaging import version

import json
import logging
import random
from glob import glob
from tqdm import tqdm
import shutil
from multiprocessing import Pool

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def download_file_cloud(url, filename):
    try:
        r = requests.get(url, stream=True)
        total_size = int(r.headers['content-length'])
        version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
        try:
            local_size = os.path.getsize(filename)
            if local_size == total_size:
                print(f'{filename} local size matched with cloud size')
                return version
        except Exception as e:
            print(e)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            for data in r.iter_content(chunk_size=1_048_576):
                f.write(data)
    except Exception as e:
        print(f'download_file_cloud error: {e}')


def get_dataset(files, directory='tfrecord', overwrite_directory=True):
    os.makedirs(directory, exist_ok=True)
    if overwrite_directory:
        shutil.rmtree(directory)
    files_to_download = []
    for f in files:
        filename = os.path.join(directory, '-'.join(f.split('/')[-2:]))
        files_to_download.append((f, filename))

    pool = Pool(processes=len(files))
    pool.starmap(download_file_cloud, files_to_download)
    pool.close()
    pool.join()
    tfrecords = glob(f'{directory}/*.tfrecord')
    return tfrecords


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.compat.v1.VarLenFeature(tf.float32),
        'targets': tf.compat.v1.VarLenFeature(tf.int64),
        'targets_length': tf.compat.v1.VarLenFeature(tf.int64),
        'lang': tf.compat.v1.VarLenFeature(tf.int64),
    }
    features = tf.compat.v1.parse_single_example(
        serialized_example, features=data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    keys = list(features.keys())
    for k in keys:
        if k not in ['waveforms', 'waveforms_length', 'targets']:
            features.pop(k, None)

    return features


class MalayaDataset(torch.utils.data.Dataset):
    def __init__(self, files, directory, batch_files=10, max_batch=999999, overwrite_directory=True,
                 start=False):
        self.files = files
        self.directory = directory
        self.batch_files = batch_files
        self.i = 0
        self.d = None
        self.sr = 16000
        self.maxlen = 12
        self.minlen = 1
        self.max_batch = max_batch
        self.overwrite_directory = overwrite_directory
        if start:
            self.get_dataset()

    def get_dataset(self, num_cpu_threads=4, thread_count=12):
        if self.i >= len(self.files) or self.i == 0:
            self.i = 0
            random.shuffle(self.files)
        b = self.files[self.i: self.i + self.batch_files]
        tfrecords = get_dataset(b, directory=self.directory, overwrite_directory=self.overwrite_directory)
        d = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecords))
        d = d.repeat(2)
        d = d.shuffle(buffer_size=len(tfrecords))
        cycle_length = min(num_cpu_threads, len(tfrecords))
        d = d.interleave(
            tf.data.TFRecordDataset,
            cycle_length=cycle_length,
            block_length=thread_count)
        d = d.shuffle(buffer_size=100)
        d = d.map(parse, num_parallel_calls=num_cpu_threads)
        d = d.filter(
            lambda x: tf.less(tf.shape(x['waveforms'])[0] / self.sr, self.maxlen)
        )
        d = d.filter(
            lambda x: tf.greater(tf.shape(x['waveforms'])[0] / self.sr, self.minlen)
        )
        self.d = d.as_numpy_iterator()
        self.i += self.batch_files

    def __getitem__(self, idx, raise_exception=False):
        try:
            r = next(self.d)
        except Exception as e:
            if raise_exception:
                raise
            print('Exception __getitem__', e)
            self.get_dataset()
            r = next(self.d)
        r = {'speech': [r['waveforms']], 'sampling_rate': 16000}
        return r

    def __len__(self):
        return self.max_batch


with open('huggingface-3mixed-train-test.json') as fopen:
    dataset = json.load(fopen)

with open('huggingface-khursani-malay.json') as fopen:
    khursani_dataset = json.load(fopen)

test_set = [
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/mandarin/0-35.tfrecord',
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/malay/2-25.tfrecord',
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/singlish/2-34.tfrecord'
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format

        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


class PretrainedTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                num_losses = inputs["mask_time_indices"].sum()
                sub_attention_mask = inputs.pop("sub_attention_mask", None)
                sub_attention_mask = (
                    sub_attention_mask if sub_attention_mask is not None else torch.ones_like(
                        inputs["mask_time_indices"])
                )
                percent_masked = num_losses / sub_attention_mask.sum()

                outputs = model(**inputs)
                loss = outputs.loss / num_losses
        else:
            num_losses = inputs["mask_time_indices"].sum()
            sub_attention_mask = inputs.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(
                    inputs["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            outputs = model(**inputs)
            loss = outputs.loss / num_losses

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[2]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # train_dataset = MalayaDataset(test_set,
    #                               directory='tfrecord-300m-test',
    #                               max_batch=100,
    #                               overwrite_directory=False)
    train_dataset = MalayaDataset(dataset['train'] + khursani_dataset, directory='tfrecord-300m')

    # 2. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)

    # only normalized-inputs-training is supported
    if not feature_extractor.do_normalize:
        raise ValueError(
            "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
        )

    def prepare_dataset(batch):

        inputs = feature_extractor(
            batch["speech"], sampling_rate=batch["sampling_rate"]
        )

        new_batch = {'input_values': inputs.input_values[0]}
        return new_batch

    train_dataset = train_dataset.map(
        prepare_dataset,
    )
    print(model_args)

    # 3. Load model
    config = Wav2Vec2Config.from_pretrained(
        model_args.model_name_or_path,
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=4,
        mask_time_prob=model_args.mask_time_prob,
    )

    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and"
            " ``config.feat_extract_norm='layer'"
        )

    # initialize random model
    model = Wav2Vec2ForPreTraining(config)

    # 4. Define data collator, optimizer and scheduler
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model, feature_extractor=feature_extractor, padding=True,
    )

    trainer = PretrainedTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=feature_extractor,
    )

    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        print('checkpoint', checkpoint)

        feature_extractor.save_pretrained(training_args.output_dir)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = 100
        metrics["eval_samples"] = 100

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


if __name__ == "__main__":
    main()
