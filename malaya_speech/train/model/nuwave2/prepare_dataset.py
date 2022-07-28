from malaya_speech.utils.read import load
import numpy as np
import random

from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import resample_poly


class Dataset:
    def __init__(self, files, hparams, cv=0, sr=24000):
        self.hparams = hparams
        self.cv = cv
        self.sr = sr
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        wav, _ = load(self.files[index], self.hparams.audio.sampling_rate)
        wav /= np.max(np.abs(wav))

        if wav.shape[0] < self.hparams.audio.length:
            padl = self.hparams.audio.length - wav.shape[0]
            r = random.randint(0, padl) if self.cv < 2 else padl // 2
            wav = np.pad(wav, (r, padl - r), 'constant', constant_values=0)
        else:
            start = random.randint(0, wav.shape[0] - self.hparams.audio.length)
            wav = wav[start:start + self.hparams.audio.length] if self.cv < 2 \
                else wav[:len(wav) - len(wav) % self.hparams.audio.hop_length]
        wav *= random.random() / 2 + 0.5 if self.cv < 2 else 1

        if self.cv == 0:
            order = random.randint(1, 11)
            ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])
            highcut = random.randint(self.hparams.audio.sr_min // 2, self.hparams.audio.sr_max // 2)
        else:
            order = 8
            ripple = 0.05
            if self.cv == 1:
                highcut = random.choice([8000 // 2, 12000 // 2, 16000 // 2, 24000 // 2])
            elif self.cv == 2:
                highcut = self.sr // 2

        nyq = 0.5 * self.hparams.audio.sampling_rate
        hi = highcut / nyq

        if hi == 1:
            wav_l = wav
        else:
            sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
            wav_l = sosfiltfilt(sos, wav)

            wav_l = resample_poly(wav_l, highcut * 2, self.hparams.audio.sampling_rate)
            # upsample to the original sampling rate
            wav_l = resample_poly(wav_l, self.hparams.audio.sampling_rate, highcut * 2)

        if len(wav_l) < len(wav):
            wav_l = np.pad(wav, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
        elif len(wav_l) > len(wav):
            wav_l = wav_l[:len(wav)]

        fft_size = self.hparams.audio.filter_length // 2 + 1
        band = np.zeros(fft_size, dtype=np.int32)
        band[:int(hi * fft_size)] = 1

        return wav.astype(np.float32), wav_l.astype(np.float32), band

    def batch(self, wavs, wav_ls, bands):

        wav_list = np.stack(wavs, axis=0)
        wav_l_list = np.stack(wav_ls, axis=0)
        band_list = np.stack(bands, axis=0)

        t = ((1 - np.random.uniform(size=(1,))) + np.arange(wav_list.shape[0]) / wav_list.shape[0]) % 1

        # tf, ((1 - tf.random.uniform(shape = (1,))) + tf.range(10,dtype=tf.float32) / 10) % 1

        return wav_list, wav_l_list, band_list, t.astype(np.float32)
