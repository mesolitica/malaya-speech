import torch
import torch.nn as nn
from .diffusion import Diffusion
from malaya_speech.config.nuwave2 import config_torch
from malaya_boilerplate.train.config import HParams


class NuWave2(nn.Module):
    def __init__(self):
        super().__init__()
        hparams = HParams(**config_torch)
        self.hparams = hparams

        self.model = Diffusion(hparams)

    @torch.no_grad()
    def inference(self, wav_l, band, step, noise_schedule=None):
        signal = torch.randn(wav_l.shape, dtype=wav_l.dtype, device=wav_l.device)
        signal_list = []
        if noise_schedule == None:
            h = (self.hparams.logsnr.logsnr_max - self.hparams.logsnr.logsnr_min) / step
        for i in range(step):
            if noise_schedule == None:
                logsnr_t = (self.hparams.logsnr.logsnr_min + i * h) * torch.ones(signal.shape[0], dtype=signal.dtype,
                                                                                 device=signal.device)
                logsnr_s = (self.hparams.logsnr.logsnr_min + (i+1) * h) * torch.ones(signal.shape[0], dtype=signal.dtype,
                                                                                     device=signal.device)
                signal, recon = self.model.denoise_ddim(signal, wav_l, band, logsnr_t, logsnr_s)
            else:
                logsnr_t = noise_schedule[i] * torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                if i == step-1:
                    logsnr_s = self.hparams.logsnr.logsnr_max * \
                        torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                else:
                    logsnr_s = noise_schedule[i+1] * \
                        torch.ones(signal.shape[0], dtype=signal.dtype, device=signal.device)
                signal, recon = self.model.denoise_ddim(signal, wav_l, band, logsnr_t, logsnr_s)
            signal_list.append(signal)
        wav_recon = torch.clamp(signal, min=-1, max=1-torch.finfo(torch.float16).eps)
        return wav_recon, signal_list
