from malaya_speech.torch_model.voicefixer.nvsr_unet import NVSR as Model
import librosa
import torch
import torch.nn as nn
import numpy as np
from malaya_speech.utils.torch_utils import to_numpy, to_tensor_cuda

EPS = 1e-9


def trim_center(est, ref):
    diff = np.abs(est.shape[-1] - ref.shape[-1])
    if est.shape[-1] == ref.shape[-1]:
        return est, ref
    elif est.shape[-1] > ref.shape[-1]:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., int(diff // 2): -int(diff // 2)], ref
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref
    else:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est, ref[..., int(diff // 2): -int(diff // 2)]
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref


class NVSR(nn.Module):
    def __init__(self, pth, vocoder_pth):
        super(NVSR, self).__init__()
        self.model = Model(vocoder_pth, channels=1)
        self.model.load_state_dict(torch.load(pth, map_location='cpu'))
        self.model.eval()

    def _find_cutoff(self, x, threshold=0.95):
        threshold = x[-1] * threshold
        for i in range(1, x.shape[0]):
            if x[-i] < threshold:
                return x.shape[0] - i
        return 0

    def _get_cutoff_index(self, x):
        stft_x = np.abs(librosa.stft(x))
        energy = np.cumsum(np.sum(stft_x, axis=-1))
        return self._find_cutoff(energy, 0.97)

    def postprocessing(self, x, out):
        # Replace the low resolution part with the ground truth
        length = out.shape[0]
        cutoffratio = self._get_cutoff_index(x)
        stft_gt = librosa.stft(x)
        stft_out = librosa.stft(out)
        stft_out[:cutoffratio, ...] = stft_gt[:cutoffratio, ...]
        out_renewed = librosa.istft(stft_out, length=length)
        return out_renewed

    def pre(self, input, cuda):
        input = to_tensor_cuda(input[None, ...], cuda)
        sp, _, _ = self.model.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.model.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return sp, mel_orig

    def get_cutoff_index_v2(self, x):
        energy = np.cumsum(np.sum(x, axis=-1))
        return self._find_cutoff(energy, 0.97)

    def add_segment_to_higher_freq(self, mel_lr, cuda):
        # mel_lr: [128, t-steps]
        size = mel_lr.size()
        mel_lr = to_numpy(mel_lr.squeeze().transpose(0, 1))
        cutoffratio = self.get_cutoff_index_v2(mel_lr)
        avg_energy = np.tile(mel_lr[cutoffratio, :], (mel_lr.shape[0], 1))
        mel_lr[cutoffratio:, ...] = 0
        avg_energy[:cutoffratio, ...] = 0
        mel_lr = mel_lr + avg_energy
        mel_lr = to_tensor_cuda(torch.Tensor(mel_lr.copy()).transpose(0, 1)[None, None, ...], cuda)

        return mel_lr

    def forward(self, x):
        cuda = next(self.parameters()).is_cuda
        with torch.no_grad():
            segment = to_tensor_cuda(torch.Tensor(x.copy())[None, ...], cuda)
            _, mel_noisy = self.pre(segment, cuda)
            denoised_mel = self.add_segment_to_higher_freq(mel_noisy, cuda)
            out = self.model.vocoder(denoised_mel, cuda=cuda)
            out, _ = trim_center(out, segment)
            out = to_numpy(out)
            out = np.squeeze(out)
            out = self.postprocessing(x, out)
        return out
