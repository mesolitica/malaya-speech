from collections import namedtuple
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import get_linear_schedule_with_warmup
from malaya_speech.utils import nemo_featurization
from malaya_speech.nemo import conv_asr
from glob import glob
from datasets import Audio
import pytorch_lightning as pl
import random
import yaml
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
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
precision = args.precision or 16
accumulate = int(args.accumulate or 1)
save_steps = int(args.save or 10000)

"""
wget https://huggingface.co/huseinzol05/nemo-titanet_large/raw/main/model_config.yaml -O titanl_model_config.yaml
wget https://huggingface.co/huseinzol05/nemo-titanet_large/resolve/main/model_weights.ckpt -O titan_model.ckpt
"""

"""
CUDA_VISIBLE_DEVICES=1 python3 lightning-titanetl-speaker-count.py --batch=32 --precision=32
"""

Batch = namedtuple("Batch", ["features", 'features_length', "targets"])

labels = [
    '0 speaker',
    '1 speaker',
    '2 speakers',
    '3 speakers',
    '4 speakers',
    '5 speakers',
    'more than 5 speakers',
]


class Dataset(torch.utils.data.IterableDataset):

    sr = 16000

    def __init__(self):
        super(Dataset).__init__()

        files = random.sample(glob('/home/husein/ssd2/LibriSpeech/*/*/*/*.flac'), 20000)
        edge_tts = random.sample(glob('/home/husein/ssd2/*-tts-wav/*.wav'), 20000)
        wavenet = random.sample(glob('/home/husein/ssd2/ms-MY-Wavenet-*/*.mp3'), 20000)
        musan_speech = glob('/home/husein/ssd2/noise/musan/speech/*/*.wav')
        vctk = random.sample(glob('/home/husein/ssd2/wav48_silence_trimmed/*/*.flac'), 20000)
        mandarin = random.sample(glob('/home/husein/ssd3/ST-CMDS-20170001_1-OS/*.wav'), 20000)

        speeches = files + edge_tts + wavenet + musan_speech + vctk + mandarin
        print('len(speeches)', len(speeches))
        random.shuffle(speeches)
        self.speeches = speeches

        mic_noise = glob('/home/husein/ssd2/noise/mic-noise/*.mp3')
        non_speech = glob('/home/husein/ssd2/noise/Nonspeech/*')
        musan_noise = glob('/home/husein/ssd2/noise/musan/noise/*/*.wav')
        musan_music = glob('/home/husein/ssd2/noise/musan/music/*/*.wav')
        noises = mic_noise + non_speech + musan_noise + musan_music
        noises = [f for f in noises if f.endswith(
            '.mp3') or f.endswith('.wav') or f.endswith('.flac')]
        noises = [f for f in noises if os.path.getsize(f) / 1e6 < 10]
        print('len(noises)', len(noises))
        random.shuffle(noises)
        self.noises = noises

        self.audio = Audio(sampling_rate=self.sr)

        self.frame_size = 300
        self.queue_size = 200
        self.repeat = 3

    def random_sampling(self, s, length):
        return augmentation.random_sampling(s, sr=self.sr, length=length)

    def read_positive(self, f):
        y = self.audio.decode_example(self.audio.encode_example(f))['array']
        y = malaya_speech.augmentation.waveform.random_pitch(y)
        y_int = malaya_speech.astype.float_to_int(y)
        vad = malaya_speech.vad.webrtc(
            minimum_amplitude=int(
                np.quantile(
                    np.abs(y_int), 0.3)))
        frames_int = malaya_speech.generator.frames(y_int, 30, self.sr, False)
        frames = malaya_speech.generator.frames(y, 30, self.sr, False)
        frames = [(frames[no], vad(frame)) for no, frame in enumerate(frames_int)]
        grouped = malaya_speech.group.group_frames(frames)
        grouped = [g[0].array for g in grouped if g[1]]
        return np.concatenate(grouped)

    def combine(self, w_samples):
        min_len = min([len(s) for s in w_samples])
        min_len = int((min_len / 16000) * 1000)
        left = np.sum([self.random_sampling(s, min_len) for s in w_samples], axis=0)
        left = left / np.max(np.abs(left))
        return left

    def __iter__(self):
        while True:
            queue = []
            while len(queue) < self.queue_size:
                try:
                    count = random.randint(0, 6)
                    if count == 0:
                        combined = random.sample(self.noises, random.randint(1, 5))
                        ys = [self.audio.decode_example(self.audio.encode_example(f))[
                            'array'] for f in combined]
                    else:
                        if count == 6:
                            count = random.randint(6, 10)
                        combined = random.sample(self.speeches, count)
                        ys = [self.read_positive(f) for f in combined]

                    if count > 5:
                        label = 'more than 5 speakers'
                    elif count > 1:
                        label = f'{count} speakers'
                    else:
                        label = f'{count} speaker'

                    n = len(combined)
                    w_samples = [
                        self.random_sampling(y, length=random.randint(500, max(10000 // n, 5000)))
                        for y in ys
                    ]

                    X = self.combine(w_samples)
                    fs = malaya_speech.generator.frames(
                        X, self.frame_size, self.sr, append_ending_trail=False)
                    for fs_ in fs:
                        queue.append((fs_.array, labels.index(label)))
                except Exception as e:
                    print(e)

            for _ in range(self.repeat):
                random.shuffle(queue)
                for r in queue:
                    yield torch.tensor(r[0], dtype=torch.float32), r[1]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        config = 'titanl_model_config.yaml'
        with open(config) as stream:
            try:
                d = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError('invalid yaml')

        d['decoder']['num_classes'] = len(labels)
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
        loaded_state_dict = torch.load('titan_model.ckpt')
        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k,
            v in zip(
                current_model_dict.keys(),
                loaded_state_dict.values())}
        self.model.load_state_dict(new_state_dict, strict=False)

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=5e-4,
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
        self.log(
            "Losses/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss


def batch(batches):

    features = torch.nn.utils.rnn.pad_sequence([b[0] for b in batches], batch_first=True)
    features_length = torch.tensor([len(b[0]) for b in batches], dtype=torch.int32)
    targets = torch.tensor([b[1] for b in batches], dtype=torch.int64)
    return Batch(features, features_length, targets)


if __name__ == '__main__':
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=batch,
        num_workers=5,
    )

    model_directory = f'titanetl-speaker-count-v2-{precision}'
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
        gradient_clip_val=1.0,
        precision=precision,
        accumulate_grad_batches=accumulate,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(
        model=model,
        train_dataloaders=loader,
        ckpt_path=ckpt_path,
    )
