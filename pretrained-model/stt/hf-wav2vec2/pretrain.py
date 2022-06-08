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

""" Pre-Training a 🤗 Wav2Vec2 model on unlabeled audio data """

"""
python3 pretrain.py \
--model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
--output_dir="./wav2vec2-pretrained" \
--mixed_precision="fp16" \
--max_train_steps="500000" \
--num_warmup_steps="32000" \
--gradient_accumulation_steps="1" \
--learning_rate="1e-4" \
--weight_decay="0.01" \
--logging_steps="100" \
--saving_steps="1000" \
--per_device_train_batch_size="10" \
--adam_beta1="0.9" \
--adam_beta2="0.98" \
--adam_epsilon="1e-06" \
--mask_time_prob="0.2"
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['WANDB_DISABLED'] = 'false'

import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass

import json
import random
import argparse
import math
import os
import requests
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import Repository
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    is_wandb_available,
    set_seed,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.utils import get_full_repo_name

from glob import glob
from tqdm import tqdm
import shutil
import logging
import sys
from multiprocessing import Pool

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

languages = defaultdict(list)

for f in dataset['train']:
    l = f.split('/')[-2]
    languages[l].append(f)

train_set = random.sample(languages['mandarin'], 650) + \
    random.sample(languages['singlish'], 650) + \
    languages['malay'] + khursani_dataset

test_set = [
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/mandarin/0-35.tfrecord',
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/malay/2-25.tfrecord',
    'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main/singlish/2-34.tfrecord'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--preprocessing_only",
        action="store_true",
        help="Only run the preprocessing script to be cached for future use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=1,
        help="Percentage of training data that should be used for validation if no validation is present in dataset.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--audio_column_name",
        type=str,
        default="audio",
        help="Column in the dataset that contains speech file path. Defaults to 'audio'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_cache_file_name",
        type=str,
        default=None,
        help="Path to the train cached file name",
    )
    parser.add_argument(
        "--validation_cache_file_name",
        type=str,
        default=None,
        help="Path to the validation cached file name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=5.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=3.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help=(
            "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the"
            " use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='no',
        choices=["no", "fp16", "bf16"],
        help="Whether or not to use mixed precision training. "
        "Choose between FP16 and BF16 (bfloat16) training. "
        "BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.",
    )
    parser.add_argument(
        "--mask_time_prob",
        type=float,
        default=0.2,
        help="Propability of each feature vector along the time axis to be chosen as the start of the vector",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


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
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
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


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

        # set up weights and biases if available
        if is_wandb_available():
            import wandb

            wandb.init(project=args.output_dir.split("/")[-1])
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub and not args.preprocessing_only:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # train_dataset = MalayaDataset(test_set,
    #                               directory='tfrecord-300m-test',
    #                               max_batch=100000,
    #                               overwrite_directory=False)
    train_dataset = MalayaDataset(train_set, directory='tfrecord-300m')

    # 2. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)

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

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            prepare_dataset,
        )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if args.preprocessing_only:
        return

    # 3. Load model
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path,
                                            mask_time_prob=args.mask_time_prob,
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

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. Define data collator, optimizer and scheduler
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model, feature_extractor=feature_extractor, pad_to_multiple_of=args.pad_to_multiple_of
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    optimizer = AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # 5. Train
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0
    starting_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # compute num of losses
            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            # forward
            outputs = model(**batch)

            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # make sure that `num_losses` is summed for distributed training
            # and average gradients over losses of all devices
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / num_losses)

            # update step
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                # update gumbel temperature
                gumbel_temperature = max(
                    args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
                    args.min_gumbel_temperature,
                )
                if hasattr(model, "module"):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)

                progress_bar.update(1)
                completed_steps += 1

            # 6. Log all results
            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather(loss).sum()
                    outputs.contrastive_loss = accelerator.gather(outputs.contrastive_loss).sum()
                    outputs.diversity_loss = accelerator.gather(outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather(percent_masked).sum()

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "constrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    if is_wandb_available():
                        wandb.log(train_logs)

            # save model every `args.saving_steps` steps
            if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

                if (args.push_to_hub and epoch < args.num_train_epochs - 1) and accelerator.is_main_process:
                    repo.push_to_hub(
                        commit_message=f"Training in progress step {completed_steps}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            # if completed steps > `args.max_train_steps` stop
            if completed_steps >= args.max_train_steps:
                break

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()
