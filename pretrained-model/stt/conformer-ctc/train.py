import numpy as np
import torch
import random
import malaya_speech
import json
import os
import pytsmod as tsm
from audiomentations import LowPassFilter
from datasets import Audio
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import AutoModel
from transformers.trainer_utils import get_last_checkpoint
from malaya_speech.utils import torch_featurization
from conformer import HF_CTC_VOCAB, ConformerConfig, ConformerEncoder
from streaming import LocalDataset

HF_CTC_VOCAB_INDEX = {no: c for no, c in enumerate(HF_CTC_VOCAB)}
HF_CTC_VOCAB_REV = {v: k for k, v in HF_CTC_VOCAB_INDEX.items()}

lowpassfilter = LowPassFilter(
    p=1.0, min_cutoff_freq=2000, max_cutoff_freq=3000,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})


@dataclass
class DataCollatorCTCWithPadding:
    def __call__(self, features):
        inputs = [f['inputs'] for f in features if f is not None]
        lengths = torch.tensor([len(f['inputs']) for f in features if f is not None])
        labels = [torch.tensor(f['labels']) for f in features if f is not None]
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'inputs': inputs,
            'lengths': lengths,
            'labels': labels,
        }


def downsample(y, sr, srs=[4400, 5100, 6000, 8000, 10000]):
    s_sr = random.choice(srs)
    y_ = malaya_speech.resample(y, sr, s_sr)
    return malaya_speech.resample(y_, s_sr, sr)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local):
            self.dataset = LocalDataset(local=local)
            self.sr = 16000
            self.audio = Audio(sampling_rate=self.sr)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            try:
                y = self.audio.decode_example(
                    self.audio.encode_example(
                        data['audio_filename']
                    )
                )['array']

                if (len(y) / self.sr) > 16:
                    return None

                if random.random() > 0.7:
                    k = random.randint(0, 1)
                    if k == 0:
                        y = downsample(y, self.sr)
                    if k == 1:
                        y = lowpassfilter(y, sample_rate=self.sr)

                if random.random() > 0.9:
                    y = tsm.wsola(y, random.uniform(0.8, 1.2))

                mel = torch_featurization.melspectrogram(y)
                mel = torch_featurization.piecewise_linear_log(mel)

                text = [HF_CTC_VOCAB_REV[c] for c in data['text']]

                return {
                    'inputs': mel,
                    'labels': text,
                }
            except Exception as e:
                print(e, data)
                return None

        def __len__(self):
            return len(self.dataset)

    dataset = DatasetFixed(data_args.train_file)

    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    print(model)
    collator = DataCollatorCTCWithPadding()

    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=dataset,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
