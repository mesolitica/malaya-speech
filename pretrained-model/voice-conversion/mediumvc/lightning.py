import pickle
from glob import glob
from datasets import Audio
from sklearn.utils import shuffle
import random
import torch
import json
from librosa.util import normalize
from torch.nn.utils.rnn import pad_sequence
from malaya_speech.augmentation.waveform import random_sampling
from malaya_speech.torch_model.hifivoice.env import AttrDict
from malaya_speech.torch_model.hifivoice.meldataset import mel_spectrogram, mel_normalize
from malaya_speech.torch_model.mediumvc.any2any import MagicModel
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', help='batch size')
parser.add_argument('-c', '--checkpoint', help='checkpoint')
parser.add_argument('-g', '--gradient_clipping', help='gradient clipping')
parser.add_argument('-p', '--precision', help='precision')
parser.add_argument('-a', '--accumulate', help='accumulate')
parser.add_argument('-s', '--save', help='save steps')
args = parser.parse_args()
batch_size = int(args.batch)
ckpt_path = args.checkpoint
precision = int(args.precision or 16)
accumulate = int(args.accumulate or 1)
save_steps = int(args.save or 5000)

"""
CUDA_VISIBLE_DEVICES=1 python3 lightning-mediumvc.py --batch=16 --precision=32
"""


class Module(LightningModule):
    def __init__(self, ):
        super().__init__()

        self.Generator = MagicModel(d_model=192)
        self.criterion = torch.nn.L1Loss()

        config = 'hifigan-config.json'
        with open(config) as fopen:
            json_config = json.load(fopen)

        self.config = AttrDict(json_config)

        # self.optimizer = torch.optim.AdamW(
        #     [{'params': filter(lambda p: p.requires_grad, self.Generator.parameters()), 'initial_lr': self.config["learning_rate"]}],
        #     self.config["learning_rate"], betas=[self.config["adam_b1"], self.config["adam_b2"]])
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizer, gamma=self.config["lr_decay"], last_epoch=-1)

        self.optimizer = torch.optim.AdamW(
            [{'params': filter(lambda p: p.requires_grad, self.Generator.parameters()),
              'initial_lr': 5e-5}], 5e-5, betas=[self.config["adam_b1"], self.config["adam_b2"]])

    def configure_optimizers(self):
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=10000,
            num_training_steps=1000000,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return (
            [self.optimizer],
            [scheduler],
        )

    def training_step(self, batch, batch_idx):
        spk_embs, input_mels, input_masks, overlap_lens = batch
        fake_mels = self.Generator(spk_embs, input_mels, input_masks)
        losses = []
        for fake_mel, target_mel, overlap_len in zip(
                fake_mels.unbind(), input_mels.unbind(), overlap_lens):
            temp_loss = self.criterion(fake_mel[:overlap_len, :], target_mel[:overlap_len, :])
            losses.append(temp_loss)
        loss = sum(losses) / len(losses)
        self.log(f"Losses/training_loss", loss, on_step=True, on_epoch=True)
        return loss


class Dataset(torch.utils.data.IterableDataset):

    sr = 22050

    def __init__(self):
        super(Dataset).__init__()

        self.speakers = glob('random-embedding-*.pkl') + \
            glob('/home/husein/ssd2/processed-youtube/*.pkl')
        self.speakers = shuffle(self.speakers)

        self.audio = Audio(sampling_rate=self.sr)
        config = 'hifigan-config.json'
        with open(config) as fopen:
            json_config = json.load(fopen)

        self.config = AttrDict(json_config)

    def __iter__(self):
        while True:
            batch = []
            for i in range(len(self.speakers)):
                with open(self.speakers[i], 'rb') as fopen:
                    data = pickle.load(fopen)

                data = random.sample(data, min(len(data), 4))
                for d in data:
                    spk_emb = d['classification_model'][0]
                    y = self.audio.decode_example(self.audio.encode_example(d['wav_data']))
                    y = y['array']
                    y = random_sampling(y, 22050, length=8000)
                    batch.append((y, spk_emb))

                if len(batch) >= 32:
                    batch = shuffle(batch)
                    for y, spk_emb in batch:
                        spk_emb = normalize(spk_emb)
                        audio = normalize(y) * 0.95
                        audio = torch.FloatTensor(audio)
                        audio = audio.unsqueeze(0)

                        mel = mel_spectrogram(audio,
                                              self.config["n_fft"],
                                              self.config["num_mels"],
                                              self.config["sampling_rate"],
                                              self.config["hop_size"],
                                              self.config["win_size"],
                                              self.config["fmin"],
                                              self.config["fmax"],
                                              center=False)

                        mel = mel.squeeze(0).transpose(0, 1)
                        mel = mel_normalize(mel)

                        yield mel, torch.tensor(spk_emb)

                    batch = []


def batch(batches):
    ori_mels, spk_input_mels = zip(*batches)

    spk_input_mels = torch.stack(spk_input_mels)
    ori_lens = [len(ori_mel) for ori_mel in ori_mels]

    overlap_lens = ori_lens
    ori_mels = pad_sequence(ori_mels, batch_first=True)
    mel_masks = [torch.arange(ori_mels.size(1)) >= mel_len for mel_len in ori_lens]
    mel_masks = torch.stack(mel_masks)  #

    return spk_input_mels, ori_mels, mel_masks, overlap_lens


if __name__ == '__main__':

    dataset = Dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=batch
    )

    model_directory = f'mediumvc-{precision}'
    model = Module()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='step',
        mode='max',
        dirpath=model_directory,
        every_n_train_steps=save_steps,
        filename='model-{epoch:02d}-{step}',
    )

    num_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',
        devices=num_gpus,
        limit_val_batches=100,
        precision=precision,
        accumulate_grad_batches=accumulate,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(
        model=model,
        train_dataloaders=loader,
        ckpt_path=ckpt_path,
    )
