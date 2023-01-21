import json
import torch
import numpy as np
import transformers
import datasets
import enchant
import whisper
from datasets import Dataset, Audio, DatasetDict
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    WhisperTokenizer,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pydub import AudioSegment
import malaya_speech
import random
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import argparse

"""
CUDA_VISIBLE_DEVICES=0 python3 whisper-lightning.py --model='openai/whisper-tiny' --batch=24 --precision=16
CUDA_VISIBLE_DEVICES=1 python3 whisper-lightning.py --model='openai/whisper-base' --batch=16 --precision=16
"""

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
precision = int(args.precision or 16)


class MalayaDataset(torch.utils.data.Dataset):

    SR = 16000
    MAXLEN = 20
    MINLEN = 0.1
    MINLEN_TEXT = 1
    do_lower_case = True

    def __init__(self, file, tokenizer):
        with open(file) as fopen:
            self.data = json.load(fopen)
        self.d = enchant.Dict('en_US')
        self.d.check('Hello')
        self.audio = Audio(sampling_rate=self.SR)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        x = self.data['X'][idx]
        y = self.data['Y'][idx]

        if '/imda' in x:
            lang = 'en'
        else:
            lang = 'ms'
            if random.random() > 0.4:
                splitted = y.split()
                for word in splitted:
                    if self.d.check(word):
                        lang = 'en'
                        break

        r = self.audio.decode_example(self.audio.encode_example(x))

        if (len(r['array']) / self.SR) > self.MAXLEN:
            return None

        if (len(r['array']) / self.SR) < self.MINLEN:
            return None

        if len(y) < self.MINLEN_TEXT:
            return None

        audio = whisper.pad_or_trim(r['array'].astype(np.float32).flatten())
        mel = whisper.log_mel_spectrogram(audio)
        r['input_features'] = mel
        r['input_length'] = len(r['input_features'])

        input_str = y.lower() if self.do_lower_case else y

        label = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(
            f'<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>{input_str}<|endoftext|>'))

        r['labels'] = label

        return r

    def __len__(self):
        return len(self.data['X'])


class Model(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 1000,
        weight_decay: float = 0.0,
        eval_splits=None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, config=self.config)

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


if __name__ == '__main__':

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language='malay', task='transcribe')

    def batch(features):
        input_features, label_features = [], []
        for i in range(len(features)):
            feature = features[i]
            if feature is not None:
                input_features.append({"input_features": feature["input_features"]})
                label_features.append({"input_ids": feature["labels"]})

        batch = feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

    train_dataset = MalayaDataset('mixed-stt-train.json', tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=3, collate_fn=batch)

    val_dataset = MalayaDataset('/home/husein/ssd1/speech-bahasa/malay-asr-test.json', tokenizer)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, collate_fn=batch)

    model_directory = f"{model_name.replace('/', '-')}-{precision}"

    model = Model(
        model_name_or_path=model_name
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor='step',
        mode='max',
        dirpath=model_directory,
        every_n_train_steps=500,
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
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
