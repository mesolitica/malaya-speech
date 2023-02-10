import torch.nn as nn
import torch
import numpy as np
from malaya_speech.model.frame import Frame
from malaya_speech.utils.torch_utils import to_tensor_cuda, to_numpy, from_log
from malaya_speech.torch_model.voicefixer.base import VoiceFixer as BaseVoiceFixer
from malaya_speech.torch_model.voicefixer.nvsr import NVSR as BaseNVSR
from malaya_speech.torch_model.nuwave2_torch.inference import NuWave2 as BaseNuWave2
from scipy.signal import resample_poly


class VoiceFixer(BaseVoiceFixer):
    def __init__(self, pth, vocoder_pth, model, name):
        super(VoiceFixer, self).__init__(pth, vocoder_pth)
        self.eval()

        self.__model__ = model
        self.__name__ = name

    def predict(self, input, remove_higher_frequency: bool = True):
        """
        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame,
            must an audio with 44100 sampling rate.
        remove_higher_frequency: bool, optional (default = True)
            Remove high frequency before neural upsampling.

        Returns
        -------
        result: np.array with 44100 sampling rate
        """

        input = input.array if isinstance(input, Frame) else input
        wav_10k = input
        cuda = next(self.parameters()).is_cuda

        res = []
        seg_length = 44100 * 30
        break_point = seg_length
        while break_point < wav_10k.shape[0] + seg_length:
            segment = wav_10k[break_point - seg_length: break_point]
            if remove_higher_frequency:
                segment = self.remove_higher_frequency(segment)

            sp, mel_noisy = self._pre(self._model, segment, cuda)
            out_model = self._model(sp, mel_noisy)
            denoised_mel = from_log(out_model['mel'])
            out = self._model.vocoder(denoised_mel, cuda)

            if torch.max(torch.abs(out)) > 1.0:
                out = out / torch.max(torch.abs(out))

            out, _ = self._trim_center(out, segment)
            res.append(out)
            break_point += seg_length

        out = torch.cat(res, -1)
        return to_numpy(out[0][0])

    def forward(self, input, remove_higher_frequency: bool = True):
        return self.predict(input=input, remove_higher_frequency=remove_higher_frequency)


class NVSR(BaseNVSR):
    def __init__(self, pth, vocoder_pth, model, name):
        super(NVSR, self).__init__(pth, vocoder_pth)
        self.eval()

        self.__model__ = model
        self.__name__ = name

    def predict(self, input):
        """
        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame,
            must an audio with 44100 sampling rate.

        Returns
        -------
        result: np.array with 44100 sampling rate
        """

        input = input.array if isinstance(input, Frame) else input
        return self.forward(input)


class NuWave2(BaseNuWave2):
    def __init__(self, pth, model, name):
        super(NuWave2, self).__init__()

        ckpt = torch.load(pth, map_location='cpu')
        self.load_state_dict(ckpt)
        self.eval()

        self.__model__ = model
        self.__name__ = name

    def predict(self, input, sr: int, steps: int = 8):
        """
        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame,
            prefer 8000, 12000, 16000 or 22050 or 44000 sampling rate.
        sr: int
            sampling rate, prefer 8000, 12000, 16000 or 22050 or 44000 sampling rate.
        steps: int, optional (default=8)
            diffusion steps.

        Returns
        -------
        result: np.array with 48k sampling rate
        """
        input = input.array if isinstance(input, Frame) else input
        wav = input
        cuda = next(self.parameters()).is_cuda
        noise_schedule = None
        highcut = sr // 2
        nyq = 0.5 * self.hparams.audio.sampling_rate
        hi = highcut / nyq

        fft_size = self.hparams.audio.filter_length // 2 + 1
        band = torch.zeros(fft_size, dtype=torch.int64)
        band[:int(hi * fft_size)] = 1

        wav_l = resample_poly(wav, self.hparams.audio.sampling_rate, sr)

        wav = torch.from_numpy(wav).unsqueeze(0)
        wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0)
        band = band.unsqueeze(0)

        wav_l = to_tensor_cuda(wav_l, cuda)
        band = to_tensor_cuda(band, cuda)

        wav_recon, wav_list = self.inference(wav_l, band, steps, noise_schedule)
        return to_numpy(wav_recon[0])

    def forward(self, input, sr: int, steps: int = 8):
        return self.predict(input=input, sr=sr, steps=steps)


class HiFiGAN(torch.nn.Module):
    def __init__(self, pth, model, name):
        super().__init__()

        self._model = torch.jit.load(pth, map_location='cpu')
        self._model.eval()

        self.__model__ = model
        self.__name__ = name

    def predict(self, input, sr: int):
        """
        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame,
            prefer 8000, 12000, 16000 or 22050 or 44000 sampling rate.
        sr: int
            sampling rate, prefer 8000, 12000, 16000 or 22050 or 44000 sampling rate.

        Returns
        -------
        result: np.array with 48k sampling rate
        """
        input = input.array if isinstance(input, Frame) else input
        wav = input.astype(np.float32)
        cuda = next(self._model.parameters()).is_cuda

        wav = torch.from_numpy(wav)
        wav = to_tensor_cuda(wav, cuda)

        return to_numpy(self._model(wav, sr))

    def forward(self, input, sr: int):
        return self.predict(input=input, sr=sr)
