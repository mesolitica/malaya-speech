import os
import json
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from datasets import Audio
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer
)
from functools import partial
from dataclasses import dataclass, field
from transformers.trainer_utils import get_last_checkpoint
from malaya_speech.torch_model import conformer
from malaya_speech.utils import torch_featurization


class ConformerRNNTModule(nn.Module):
    def __init__(self, model, sp_model):
        super().__init__()

        self.sp_model = sp_model
        spm_vocab_size = self.sp_model.get_piece_size()
        self.blank_idx = spm_vocab_size

        self.model = model()
        self.loss = torchaudio.transforms.RNNTLoss(reduction='mean')

    def forward(self, features, feature_lens, targets, target_lens):
        prepended_targets = targets.new_empty(
            [targets.size(0), targets.size(1) + 1])
        prepended_targets[:, 1:] = targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = target_lens + 1
        output, src_lengths, _, _ = self.model(
            features,
            feature_lens,
            prepended_targets,
            prepended_target_lengths,
        )
        loss = self.loss(output, targets, src_lengths, target_lens)
        return ((loss,))


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class MalayaDataset(torch.utils.data.Dataset):

    SR = 16000
    MAXLEN = 16
    MINLEN = 0.1
    MINLEN_TEXT = 1

    def __init__(self, file, sp_model, stats_file):
        with open(file) as fopen:
            self.data = json.load(fopen)
        self.sp_model = sp_model
        self.global_stats = torch_featurization.GlobalStatsNormalization(stats_file)
        self.audio = Audio(sampling_rate=self.SR)

    def __getitem__(self, idx):
        x = self.data['X'][idx]
        y = self.data['Y'][idx]

        r = self.audio.decode_example(self.audio.encode_example(x))

        if (len(r['array']) / self.SR) > self.MAXLEN:
            return

        if (len(r['array']) / self.SR) < self.MINLEN:
            return

        if len(y) < self.MINLEN_TEXT:
            return

        mel = torch_featurization.melspectrogram(r['array'])
        mel = torch_featurization.piecewise_linear_log(mel)
        mel = self.global_stats(mel)

        y_ = self.sp_model.encode(y.lower())

        batch = {
            'features': mel,
            'feature_lens': len(mel),
            'targets': y_,
            'target_lens': len(y_)
        }

        return batch

    def __len__(self):
        return len(self.data['X'])


@dataclass
class ScriptArguments:

    train_dataset: str
    val_dataset: str
    model: str = field(default='conformer_rnnt_base')
    sp_model: str = field(default='../../prepare-stt/vocab/malay-stt.model')
    stats_file: str = field(default='../../prepare-stt/stats/malay-stats.json')
    worker_size: int = field(default=5)


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    print(script_args)
    print(training_args)

    sp_model = spm.SentencePieceProcessor(model_file=script_args.sp_model)
    train_data_pipeline = torch.nn.Sequential(
        FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        torchaudio.transforms.FrequencyMasking(20),
        torchaudio.transforms.FrequencyMasking(20),
        torchaudio.transforms.TimeMasking(80, p=0.2),
        torchaudio.transforms.TimeMasking(80, p=0.2),
        FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
    )

    def batch(batches):
        features, feature_lens, targets, target_lens = [], [], [], []
        for i in range(len(batches)):
            b = batches[i]
            if b is not None:
                features.append(b['features'])
                feature_lens.append(b['feature_lens'])
                targets.append(b['targets'])
                target_lens.append(b['target_lens'])

        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        features = train_data_pipeline(features)
        feature_lens = torch.tensor(feature_lens, dtype=torch.int32)

        target_lens = torch.tensor(target_lens).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)

        return {
            'features': features,
            'feature_lens': feature_lens,
            'targets': targets,
            'target_lens': target_lens
        }

    train_dataset = MalayaDataset(script_args.train_dataset, sp_model, script_args.stats_file)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=script_args.worker_size,
    )

    val_dataset = MalayaDataset(script_args.val_dataset, sp_model, script_args.stats_file)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_args.per_device_eval_batch_size,
    )

    model = ConformerRNNTModule(getattr(conformer, script_args.model), sp_model)
    signature = inspect.signature(model.forward)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=batch,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
