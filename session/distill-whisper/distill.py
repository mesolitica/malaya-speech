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
import evaluate
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader, Dataset
from datasets import Audio
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import (
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    get_scheduler,
    set_seed,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"})
    teacher_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained teacher model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    subfolder: str = field(
        default="", metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."}, )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )


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
    eval_dataset_name: str = field(
        default=None, metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training "
            "dataset name if unspecified. Load multiple evaluation datasets by separating dataset "
            "ids by a '+' symbol."}, )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_label_length: int = field(
        default=384,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={
            "help": "The number of processes to use for the preprocessing if using non-streaming mode."}, )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
            "This argument should be set for multilingual distillation only. For English speech recognition, it should be left as `None`."
        },
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )


@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={
            "help": (
                "Whether to freeze the entire encoder model. Only recommended when the entire encoder has been "
                "copied from the teacher model.")}, )
    temperature: Optional[float] = field(
        default=2.0, metadata={
            "help": "Temperature to anneal the logits when computing the softmax."})
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32", metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision).")}, )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        decoder_prev_token_id (:obj: `int`)
            The start-of-prompt token id of the decoder
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    norm_pred_str: List[str],
    norm_label_str: List[str],
    step: int,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]]
                    for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data[:num_lines],
            step=step,
        )

        # log incorrect normalised predictions
        str_data = np.asarray(str_data)
        str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"incorrect_predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data_incorrect[:num_lines],
            step=step,
        )


def get_layers_to_supervise(student_layers: int, teacher_layers: int) -> Dict:
    """Helper function to map the student layer i to the teacher layer j whose output we'd like them to emulate. Used
    for MSE loss terms in distillation (hidden-states and activations). Student layers are paired with teacher layers
    in equal increments, e.g. for a 12-layer model distilled to a 3-layer model, student layer 0 emulates teacher layer
    3 (such that it behaves like the first 4 teacher layers), student layer 1 emulates teacher layer 7, and student layer
    2 emulates teacher layer 11. This mapping is summarised by the dictionary: {0: 3, 1: 7, 2: 11}, which is precisely
    the output of this function for the arguments (student_layers=3, teacher_layers=12)."""
    layer_intervals = np.linspace(
        teacher_layers // student_layers - 1,
        teacher_layers - 1,
        student_layers,
        dtype=int)
    layer_intervals[-1] = teacher_layers - 1
    layer_map = {}

    for student_layer, teacher_layer in enumerate(layer_intervals):
        layer_map[student_layer] = teacher_layer

    return layer_map


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(
        f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None,
                       checkpoint_prefix="checkpoint") -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(
        output_dir=output_dir,
        checkpoint_prefix=checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder, max(
            checkpoints, key=lambda x: int(
                _RE_CHECKPOINT.search(x).groups()[0])))


def get_parameter_names(model, forbidden_layer_types, forbidden_module=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
    (e.g. if the module is frozen).
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser(
        (ModelArguments,
         DataTrainingArguments,
         DistillationTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    # We will let the accelerator handle device placement for us in this example
    # We simply have to specify the training precision and any trackers being used
    # We'll use the same dtype arguments as our JAX/Flax training script and convert
    # it to accelerate format
    # The teacher model can safely be cast to the dtype of training since we don't
    # update the params
    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )

    accelerator.init_trackers(project_name=data_args.wandb_project)

    # 3. Set-up basic logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    # 4. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # 5. Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(
                training_args.output_dir,
                clone_from=repo_id,
                token=training_args.hub_token)

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # set seed for determinism
    set_seed(training_args.seed)

    # 7. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    # override timestamp tokens until tokenizer issues are fixed in transformers
    timestamps = [
        AddedToken(
            "<|%.2f|>" % (i * 0.02),
            lstrip=False,
            rstrip=False) for i in range(
            1500 + 1)]
    tokenizer.add_tokens(timestamps)

    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        torch_dtype=teacher_dtype,
        use_flash_attention_2=True,
    )

    student_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=teacher_dtype,
        use_flash_attention_2=True,
    )

    if student_model.config.decoder_start_token_id is None or teacher_model.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for both the "
            f"student and teacher model. Got {student_model.config.decoder_start_token_id} for the "
            f"student and {teacher_model.config.decoder_start_token_id} for the teacher."
        )

    share_hidden_states = training_args.freeze_encoder and student_model.config.d_model == teacher_model.config.d_model

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()

    # freeze student encoder if necessary
    if training_args.freeze_encoder:
        student_model.freeze_encoder()
        student_model.model.encoder.gradient_checkpointing = False

    # if share_hidden_states:
    # tie the weights for the student encoder if we're freezing it and it's the same as the teacher
    #    student_model.model.encoder = teacher_model.model.encoder

    tokenizer.set_prefix_tokens(
        task=data_args.task,
        predict_timestamps=True
    )
    student_model.generation_config.update(
        **{
            "task": data_args.task,
        }
    )

    # 8. Create a single speech processor - make sure all processes wait until data is saved
    if accelerator.is_main_process:
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        # save the config and generation config as well
        config.save_pretrained(training_args.output_dir)
        student_model.generation_config.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()
    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    # 9. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    sampling_rate = feature_extractor.sampling_rate

    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else student_model.config.max_length
    )

    decoder_start_token_id = student_model.config.decoder_start_token_id  # <|startoftranscript|>
    decoder_prev_token_id = tokenizer.all_special_ids[-3]  # <|startofprev|>

    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers

    class Train(Dataset):
        def __init__(self, file):
            self.data = []
            with open(file) as fopen:
                for l in fopen:
                    self.data.append(json.loads(l))

            self.audio = Audio(sampling_rate=16000)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            audio = self.audio.decode_example(
                self.audio.encode_example(
                    self.data[item]['audio_filename']))['array']
            inputs = feature_extractor(audio, sampling_rate=sampling_rate)
            if self.data[item]['score_ms'] >= self.data[item]['score_en']:
                input_str = self.data[item]['predict_ms']
            else:
                input_str = self.data[item]['predict_en']

            token_ids = tokenizer(input_str, add_special_tokens=False).input_ids

            return {
                'input_features': inputs.input_features[0],
                'input_length': [len(audio)],
                'labels': token_ids,
            }

    train_dataset = Train(data_args.train_dataset_name)
    eval_dataset = Train(data_args.eval_dataset_name)

    # 11. Define Evaluation Metrics
    def compute_metrics(preds, labels):
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
            decode_with_timestamps=return_timestamps)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels,
        # so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero
        # normalized references:
        norm_pred_str = [
            norm_pred_str[i] for i in range(
                len(norm_pred_str)) if len(
                norm_label_str[i]) > 0]
        norm_label_str = [
            norm_label_str[i] for i in range(
                len(norm_label_str)) if len(
                norm_label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)
        return {"wer": wer, "wer_ortho": wer_ortho}, pred_str, label_str, norm_pred_str, norm_label_str

    # 12. Define Training Schedule
    # Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(train_dataset) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as
        # necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    if training_args.eval_steps is None:
        logger.info(
            f"eval_steps is not set, evaluating at the end of {'each epoch' if not data_args.streaming else 'training'}"
        )
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # 13. Define optimizer, LR scheduler, collator
    decay_parameters = get_parameter_names(
        student_model,
        [nn.LayerNorm],
        forbidden_module=[student_model.model.encoder] if training_args.freeze_encoder else None,
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [{"params": [param for name,
                                                param in student_model.named_parameters() if name in decay_parameters],
                                     "weight_decay": training_args.weight_decay,
                                     },
                                    {"params": [param for name,
                                                param in student_model.named_parameters() if name not in decay_parameters],
                                     "weight_decay": 0.0,
                                     },
                                    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for
    # this in warmup / total steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        decoder_prev_token_id=decoder_prev_token_id,
        input_padding="longest",
        target_padding="longest",
        max_target_length=max_label_length,
    )

    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(student_model.generation_config, "num_beams", 1)
    )

    return_timestamps = True

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": return_timestamps,
    }
    if hasattr(teacher_model.generation_config,
               "is_multilingual") and teacher_model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "task": data_args.task,
            }
        )

    # 15. Prepare everything with accelerate
    student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(
        student_model, teacher_model, optimizer, lr_scheduler
    )

    def kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence

    # Define gradient update step fn
    def train_step(
        batch,
        temperature=2.0,
    ):
        student_model.train()
        teacher_model.eval()

        student_outputs = student_model(**batch)
        with torch.no_grad():
            if share_hidden_states:
                # if the student and teacher share the same frozen encoder then we don't have to recompute the
                # encoder hidden-states for the teacher model, we can just re-use from the student
                encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state)
                teacher_outputs = teacher_model(
                    encoder_outputs=encoder_outputs, labels=batch["labels"])
            else:
                # do the full forward pass for the teacher model (encoder + decoder)
                teacher_outputs = teacher_model(**batch)

        # CE (data) loss
        ce_loss = student_outputs.loss
        # rescale distribution by temperature to ensure gradients scale correctly
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
        # log softmax of student predictions for numerical stability
        student_distribution = nn.functional.log_softmax(
            student_outputs.logits / temperature, dim=-1)
        # KL-divergence loss (scaled by temperature)
        kl_loss = kl_divergence(
            teacher_distribution,
            student_distribution,
            batch["labels"]) * temperature**2

        # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return loss, metrics

    # Define eval fn
    def eval_step(batch):
        student_model.eval()
        teacher_model.eval()

        with torch.no_grad():
            student_outputs = student_model(**batch)
            if share_hidden_states:
                encoder_outputs = BaseModelOutput(student_outputs.encoder_last_hidden_state)
                teacher_outputs = teacher_model(
                    encoder_outputs=encoder_outputs, labels=batch["labels"])
            else:
                teacher_outputs = teacher_model(**batch)

        # CE (data) loss
        ce_loss = student_outputs.loss

        # log softmax / softmax for numerical stability
        student_distribution = nn.functional.log_softmax(student_outputs.logits, dim=-1)
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits, dim=-1)
        # temperature is always 1 for eval
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"])

        # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return metrics

    def generate_step(batch):
        student_model.eval()
        output_ids = accelerator.unwrap_model(student_model).generate(
            batch["input_features"], **gen_kwargs)
        output_ids = accelerator.pad_across_processes(
            output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    logger.info("***** Running training *****")
    logger.info(
        f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info("  Instantaneous batch size per device ="
                f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps),
        desc="Train steps ... ",
        position=0,
        disable=not accelerator.is_local_main_process)
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
    else:
        resume_step = None

    for epoch in range(epochs_trained, num_epochs):
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(
                train_dataloader,
                "dataset") and isinstance(
                train_dataloader.dataset,
                IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        for batch in train_dataloader:
            with accelerator.accumulate(student_model):
                loss, train_metric = train_step(batch, temperature=training_args.temperature)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        student_model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(
                        training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    accelerator.save_state(output_dir=intermediate_dir)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(
                            training_args.save_total_limit,
                            output_dir=training_args.output_dir)

                        if cur_step == total_train_steps:
                            student_model = accelerator.unwrap_model(student_model)
                            student_model.save_pretrained(training_args.output_dir)

                        if training_args.push_to_hub:
                            repo.push_to_hub(
                                commit_message=f"Saving train state of step {cur_step}",
                                blocking=False,
                            )

                if training_args.do_eval and (cur_step %
                                              eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    student_model.eval()
                    eval_split = 'eval'
                    # ======================== Evaluating ==============================
                    eval_metrics = []
                    eval_preds = []
                    eval_labels = []
                    eval_start = time.time()

                    validation_dataloader = DataLoader(
                        eval_dataset,
                        collate_fn=data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=dataloader_num_workers,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    for batch in tqdm(
                        validation_dataloader,
                        desc=f"Evaluating {eval_split}...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Model forward
                        eval_metric = eval_step(batch)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metrics.append(eval_metric)

                        # generation
                        if training_args.predict_with_generate:
                            generated_ids = generate_step(batch)
                            # Gather all predictions and targets
                            generated_ids, labels = accelerator.gather_for_metrics(
                                (generated_ids, batch["labels"])
                            )
                            eval_preds.extend(generated_ids)
                            eval_labels.extend(labels)

                        eval_time = time.time() - eval_start
                        # normalize eval metrics
                        eval_metrics = {key: torch.mean(torch.stack(
                            [d[key] for d in eval_metrics])) for key in eval_metrics[0]}

                        # compute WER metric
                        wer_desc = ""
                        if training_args.predict_with_generate:
                            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(
                                eval_preds, eval_labels)
                            eval_metrics.update(wer_metric)
                            wer_desc = " ".join(
                                [f"Eval {key}: {value} |" for key, value in wer_metric.items()])
                            log_pred(
                                accelerator,
                                pred_str,
                                label_str,
                                norm_pred_str,
                                norm_label_str,
                                step=cur_step,
                                prefix=eval_split,
                            )

                        # Print metrics and update progress bar
                        steps_trained_progress_bar.write(
                            f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                            f" {wer_desc})")

                        log_metric(
                            accelerator,
                            metrics=eval_metrics,
                            train_time=eval_time,
                            step=cur_step,
                            epoch=epoch,
                            prefix=eval_split,
                        )

                    # flush the train metrics
                    train_start = time.time()

                # break condition
                if cur_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
