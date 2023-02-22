from collections import namedtuple

import torch
from glob import glob
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from datasets import Audio
import random
from transformers import get_linear_schedule_with_warmup
import yaml
from malaya_speech.utils import nemo_featurization
from malaya_speech.nemo import conv_asr
import malaya_speech
import numpy as np
import torch
import argparse
import os

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
save_steps = int(args.save or 10000)

"""
CUDA_VISIBLE_DEVICES=1 python3 lightning-speakernet-is-clean.py --batch=32 --precision=32
"""

Batch = namedtuple("Batch", ["features", 'features_length', "targets"])


class Dataset(torch.utils.data.IterableDataset):

    sr = 16000

    def __init__(self):
        super(Dataset).__init__()

        files = random.sample(glob('/home/husein/ssd2/LibriSpeech/*/*/*/*.flac'), 10000)
        edge_tts = random.sample(glob('/home/husein/ssd2/*-tts-wav/*.wav'), 10000)
        wavenet = random.sample(glob('/home/husein/ssd2/ms-MY-Wavenet-*/*.mp3'), 10000)
        musan_speech = glob('/home/husein/ssd2/noise/musan/speech/*/*')
        vctk = random.sample(glob('/home/husein/ssd2/wav48_silence_trimmed/*/*.flac'), 10000)

        speeches = files + edge_tts + wavenet + musan_speech + vctk
        random.shuffle(speeches)
        self.speeches = speeches

        mic_noise = glob('/home/husein/ssd2/noise/mic-noise/*')
        non_speech = glob('/home/husein/ssd2/noise/Nonspeech/*')
        musan_noise = glob('/home/husein/ssd2/noise/musan/noise/*/*.wav')
        musan_music = glob('/home/husein/ssd2/noise/musan/music/*/*.wav')
        noises = mic_noise + non_speech + musan_noise + musan_music
        noises = [f for f in noises if os.path.getsize(f) / 1e6 < 10]
        random.shuffle(noises)
        self.noises = noises

        self.audio = Audio(sampling_rate=self.sr)

        self.frame_sizes = [100, 200, 300]

    def __iter__(self):
        while True:
            for i in range(len(self.speeches)):
                f = self.speeches[i]
                try:

                    y = self.audio.decode_example(self.audio.encode_example(f))['array']
                    y = malaya_speech.augmentation.waveform.random_pitch(y)
                    noise = self.audio.decode_example(
                        self.audio.encode_example(
                            random.choice(self.noises)))['array']
                    y_noise = malaya_speech.augmentation.waveform.add_noise(
                        y, noise, factor=random.uniform(0.3, 0.7))

                    y_int = malaya_speech.astype.float_to_int(y)

                    vad = malaya_speech.vad.webrtc(
                        minimum_amplitude=int(
                            np.quantile(
                                np.abs(y_int), 0.3)))
                    frames_int = malaya_speech.generator.frames(y_int, 30, self.sr, False)
                    frames = malaya_speech.generator.frames(y, 30, self.sr, False)
                    frames_noise = malaya_speech.generator.frames(y_noise, 30, self.sr, False)
                    frames = [(frames[no], vad(frame)) for no, frame in enumerate(frames_int)]
                    frames_noise = [(frames_noise[no], vad(frame))
                                    for no, frame in enumerate(frames_int)]
                    grouped = malaya_speech.group.group_frames(frames)
                    grouped_noise = malaya_speech.group.group_frames(frames_noise)

                except Exception as e:
                    continue

                if len(grouped) != len(grouped_noise):
                    continue

                frame_size = random.choice(self.frame_sizes)

                results = []
                for k in range(len(grouped)):
                    fs = malaya_speech.generator.frames(grouped[k][0], frame_size, self.sr, True)
                    for fs_ in fs:
                        results.append((fs_.array, int(grouped[k][1])))

                    fs = malaya_speech.generator.frames(
                        grouped_noise[k][0], frame_size, self.sr, True)
                    for fs_ in fs:
                        results.append((fs_.array, 0))

                random.shuffle(results)

                for r in results:
                    yield torch.tensor(r[0], dtype=torch.float32), r[1]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        config = 'model_config.yaml'
        with open(config) as stream:
            try:
                d = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError('invalid yaml')

        d['decoder']['num_classes'] = 2
        d['decoder']['angular'] = False

        preprocessor = d['preprocessor'].copy()
        preprocessor_target = (preprocessor.pop('_target_', None)
                               or preprocessor.pop('cls', None)).split('.')[-1]
        if 'params' in preprocessor:
            preprocessor = preprocessor['params']

        encoder = d['encoder'].copy()
        encoder_target = (encoder.pop('_target_', None) or encoder.pop('cls', None)).split('.')[-1]
        if 'params' in encoder:
            encoder = encoder['params']

        decoder = d['decoder'].copy()
        decoder_target = (decoder.pop('_target_', None) or decoder.pop('cls', None)).split('.')[-1]
        if 'params' in decoder:
            decoder = decoder['params']

        self.preprocessor = getattr(nemo_featurization, preprocessor_target)(**preprocessor)
        self.encoder = getattr(conv_asr, encoder_target)(**encoder)
        self.decoder = getattr(conv_asr, decoder_target)(**decoder)

    def forward(self, inputs, lengths):

        o_processor = self.preprocessor(inputs, lengths)
        o_encoder = self.encoder(*o_processor)
        return self.decoder(*o_encoder)


class Module(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Model()

        current_model_dict = self.model.state_dict()
        loaded_state_dict = torch.load('model_weights.ckpt')
        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k,
            v in zip(
                current_model_dict.keys(),
                loaded_state_dict.values())}
        self.model.load_state_dict(new_state_dict, strict=False)

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=9e-5,
            betas=(
                0.9,
                0.98),
            eps=1e-9)

    def configure_optimizers(self):
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=30000,
            num_training_steps=1000000,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return (
            [self.optimizer],
            [scheduler],
        )

    def training_step(self, batch: Batch, batch_idx):
        out = self.model(batch.features, batch.features_length)[0]
        loss = self.loss(out, batch.targets)
        self.log(f"Losses/train_loss", loss, on_step=True, on_epoch=True)

        return loss


def batch(batches):

    features = torch.nn.utils.rnn.pad_sequence([b[0] for b in batches], batch_first=True)
    features_length = torch.tensor([len(b[0]) for b in batches], dtype=torch.int32)
    targets = torch.tensor([b[1] for b in batches], dtype=torch.int64)
    return Batch(features, features_length, targets)


if __name__ == '__main__':
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=batch)

    model_directory = f'speakernet-is-clean-{precision}'
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
