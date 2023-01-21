import json
import torch
import numpy as np
import transformers
import datasets
from datasets import Dataset, Audio, DatasetDict
from transformers import (
    Wav2Vec2CTCTokenizer,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
    AutoModelForCTC,
)
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pydub import AudioSegment
import malaya_speech
import random
import string
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, LoudnessNormalization, RoomSimulator
from audiomentations.augmentations.room_simulator import RoomSimulator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name')
parser.add_argument('-b', '--batch', help='batch size')
parser.add_argument('-c', '--checkpoint', help='checkpoint')
parser.add_argument('-l', '--learning_rate', help='learning rate')
parser.add_argument('-g', '--gradient_clipping', help='gradient clipping')
parser.add_argument('-p', '--precision', help='precision')
args = parser.parse_args()
model_name = args.model
batch_size = int(args.batch)
ckpt_path = args.checkpoint
learning_rate = float(args.learning_rate or 2e-5)
gradient_clipping = float(args.gradient_clipping or 1.0)
precision = int(args.precision or 16)


def check_nan(a):
    return np.isnan(np.sum(a))


def check_inf(a):
    return np.isinf(np.sum(a))


class MalayaDataset(torch.utils.data.Dataset):

    SR = 16000
    MAXLEN = 16
    MINLEN = 0.1
    MINLEN_TEXT = 1

    def __init__(self, file, model_name_or_path: str, feature_extractor, tokenizer, use_hf_audio=True):
        with open(file) as fopen:
            self.data = json.load(fopen)
        self.use_hf_audio = use_hf_audio
        self.audio = Audio(sampling_rate=self.SR)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            RoomSimulator(p=0.5),
        ])

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

        if random.random() > 0.6:
            try:
                a = self.augment(samples=r['array'], sample_rate=r['sampling_rate'])
                if not check_nan(a):
                    r['array'] = a
                if not check_inf(a):
                    r['array'] = a
            except Exception as e:
                print(e)

        inputs = self.feature_extractor(r['array'], sampling_rate=r['sampling_rate'])

        if check_nan(inputs.input_values[0]):
            return

        if check_inf(inputs.input_values[0]):
            return

        batch = {}
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = self.tokenizer(y).input_ids

        return batch

    def __len__(self):
        return len(self.data['X'])


class Model(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-6,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        eval_splits=None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.update(
            {
                "feat_proj_dropout": 0.0,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "final_dropout": 0.0,
                "mask_time_prob": 0.05,
                "mask_time_length": 10,
                "mask_feature_prob": 0.0,
                "mask_feature_length": 10,
                "layerdrop": 0.0,
                "ctc_loss_reduction": 'mean',
                "pad_token_id": tokenizer.pad_token_id,
                "vocab_size": len(tokenizer),
                "activation_dropout": 0.0,
            }
        )
        self.model = AutoModelForCTC.from_pretrained(
            model_name_or_path, config=self.config
        )

        self.model.freeze_feature_encoder()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch=batch, batch_idx=batch_idx)

        return {"loss": loss}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
                          betas=(0.9, 0.98))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


if __name__ == '__main__':

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab-ctc.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    def batch(features):
        input_features, label_features = [], []
        for i in range(len(features)):
            feature = features[i]
            if feature is not None:
                input_features.append({"input_values": feature["input_values"]})
                label_features.append({"input_ids": feature["labels"]})

        batch = feature_extractor.pad(
            input_features,
            padding='longest',
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        labels_batch = tokenizer.pad(
            label_features,
            padding='longest',
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"]

        return batch

    model_directory = model_name.replace('/', '-')

    train_dataset = MalayaDataset('mixed-stt-train.json', model_name, feature_extractor, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=5, collate_fn=batch)

    val_dataset = MalayaDataset('/home/husein/ssd1/speech-bahasa/malay-asr-test.json',
                                model_name, feature_extractor, tokenizer)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, num_workers=2, collate_fn=batch)

    model = Model(
        model_name_or_path=model_name, tokenizer=tokenizer
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor='step',
        mode='max',
        dirpath=model_directory,
        every_n_train_steps=2000,
        filename='model-{epoch:02d}-{step}',
    )
    num_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=num_gpus,
        limit_val_batches=100,
        precision=precision,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=gradient_clipping,
        track_grad_norm=2,
    )

    print(trainer.__dict__)

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
