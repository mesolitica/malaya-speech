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
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from datasets import load_dataset
import transformers
import random
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from transformers import AddedToken
import streaming
import json
import numpy as np
import pandas as pd
from cut_cross_entropy.transformers import cce_patch

cce_patch("llama")
IGNORE_INDEX = -100

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def _uniform_assignment(src_lens, tgt_lens):
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)
    ratio = tgt_lens / src_lens
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t

class SpeechGeneratorCTC(torch.nn.Module):
    def __init__(self, config, ctc_upsample_factor = 26, unit_vocab_size = 1024):
        super().__init__()
        n_layers, n_dims, n_heads, n_inter_dims = 2,4096,32,11008
        _config = copy.deepcopy(config)
        _config.hidden_size = n_dims
        _config.num_hidden_layers = n_layers
        _config.num_attention_heads = n_heads
        _config.num_key_value_heads = n_heads
        _config.intermediate_size = n_inter_dims
        _config._attn_implementation = "flash_attention_2"
        self.upsample_factor = ctc_upsample_factor
        self.input_proj = nn.Linear(config.hidden_size, n_dims)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(_config, layer_idx) for layer_idx in range(n_layers)]
        )
        self.unit_vocab_size = unit_vocab_size
        self.output_proj = nn.Linear(n_dims, self.unit_vocab_size + 1)
    
    def upsample(self, reps, tgt_units=None):
        src_lens = torch.LongTensor([len(rep) for rep in reps]).to(reps[0].device)
        up_lens = src_lens * self.upsample_factor
        if tgt_units is not None:
            tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
            up_lens = torch.max(up_lens, tgt_lens)
        reps = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True)
        padding_mask = lengths_to_padding_mask(up_lens)
        mapped_inputs = _uniform_assignment(src_lens, up_lens).masked_fill(
            padding_mask, 0
        )
        copied_reps = torch.gather(
            reps,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), reps.size(-1)
            ),
        )
        copied_reps = copied_reps.masked_fill(padding_mask.unsqueeze(-1), 0)
        position_ids = torch.arange(0, max(up_lens)).unsqueeze(0).expand(len(reps), -1).to(device=copied_reps.device)
        return copied_reps, ~padding_mask, position_ids

    def forward(self, tgt_reps, labels, tgt_units):
        tgt_label_reps = []
        for tgt_rep, label in zip(tgt_reps, labels):
            tgt_label_reps.append(tgt_rep[label != IGNORE_INDEX])
        hidden_states, attention_mask, position_ids = self.upsample(tgt_label_reps, tgt_units)
        hidden_states = self.input_proj(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
        ctc_logits = self.output_proj(hidden_states)
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_lens = attention_mask.long().sum(dim=-1)
        ctc_tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
        ctc_tgt_mask = ~lengths_to_padding_mask(ctc_tgt_lens)
        ctc_tgt_flat = tgt_units.masked_select(ctc_tgt_mask)
        ctc_loss = F.ctc_loss(
            ctc_lprobs.transpose(0, 1),
            ctc_tgt_flat,
            ctc_lens,
            ctc_tgt_lens,
            reduction="sum",
            zero_infinity=True,
            blank=self.unit_vocab_size
        )
        ctc_loss /= ctc_tgt_lens.sum().item()
        return ctc_loss
    
    def predict(self, tgt_reps):
        hidden_states, attention_mask, position_ids = self.upsample([tgt_reps])
        hidden_states = self.input_proj(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
        ctc_logits = self.output_proj(hidden_states)
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_pred = ctc_lprobs.argmax(dim=-1).masked_fill_(~attention_mask, self.unit_vocab_size)
        return ctc_pred

class LlamaTTS(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.speech_generator = SpeechGeneratorCTC(self.config)
        
    def forward(self, tgt_units = None, **kwargs):
        super_out = super().forward(**kwargs, output_hidden_states = True)
        tgt_reps = super_out.hidden_states[-1]
        ctc_loss = self.speech_generator(tgt_reps, kwargs.get('labels'), tgt_units)
        return {'loss': super_out.loss + ctc_loss}

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
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " +
            ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")}, )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine.")}, )
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
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None, metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}, )
    max_train_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set.")}, )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set.")}, )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def block_diagonal_concat_inverted(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    inverted_mask = torch.where(combined_mask == 1, torch.tensor(0, dtype=dtype), min_value)
    return inverted_mask.unsqueeze(0)

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

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34.",
            FutureWarning)
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

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

    print(model_args)

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

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    new = ['<|speaker|>']
    new = [AddedToken(t) for t in new]
    tokenizer.add_tokens(new)

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, file):
            self.df = pd.read_parquet(file).to_dict(orient = 'records')
        
        def __getitem__(self, idx):
            row = self.df[idx]
            speaker = f"<|speaker|>{row['speaker']}<|speaker|>"
            prompt = f"{speaker}{row['transcription']}"
            input_ids = tokenizer(prompt, add_special_tokens = False)
            splitted = row['audio_filename'].split('/')
            new_f = '/'.join([splitted[0] + '_vqgan'] + splitted[1:]).replace('.mp3', '.npy')
            speech_token = np.load(new_f)
            input_ids['tgt_units'] = torch.tensor(speech_token)
            if input_ids['tgt_units'].shape[-1] >= (model.config.max_position_embeddings // 2):
                print(input_ids['tgt_units'].shape[-1], 'too long, skip')
                return None
            return input_ids

        def __len__(self):
            return len(self.df)

    def collator(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [{'input_ids': b['input_ids']} for b in batch]
        tgt_units = [b['tgt_units'] for b in batch]
        tgt_units = torch.nn.utils.rnn.pad_sequence(tgt_units, batch_first = True, padding_value = -100)
        input_ids = tokenizer.pad(input_ids, return_tensors = 'pt')
        input_ids['labels'] = torch.clone(input_ids['input_ids'])
        input_ids['labels'][input_ids['labels'] == tokenizer.pad_token_id] = -100
        input_ids['tgt_units'] = tgt_units
        print(input_ids['tgt_units'].shape)
        
        return input_ids

    dataset = DatasetFixed(data_args.train_file)
    print(len(dataset))

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = LlamaTTS.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype = torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
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
