import torch.utils
import torch.nn as nn
import torch.utils.data
from malaya_speech.torch_model.voicefixer.vocoder.base import Vocoder
from malaya_speech.torch_model.voicefixer.tools.modules.fDomainHelper import FDomainHelper
from malaya_speech.torch_model.voicefixer.tools.mel_scale import MelScale
from malaya_speech.torch_model.voicefixer.restorer.model_kqq_bn import UNetResComplex_100Mb
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class BN_GRU(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer=1,
        bidirectional=False,
        batchnorm=True,
        dropout=0.0,
    ):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def forward(self, inputs):
        # (batch, 1, seq, feature)
        if self.batchnorm:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class NVSR(nn.Module):
    def __init__(self, vocoder_pth, channels):
        super(NVSR, self).__init__()

        model_name = "unet"

        self.channels = channels

        self.vocoder = Vocoder(vocoder_pth, sample_rate=44100)

        self.downsample_ratio = 2**6  # This number equals 2^{#encoder_blcoks}

        self.f_helper = FDomainHelper(
            window_size=2048,
            hop_size=441,
            center=True,
            pad_mode="reflect",
            window="hann",
            freeze_parameters=True,
        )

        self.mel = MelScale(n_mels=128, sample_rate=44100, n_stft=2048 // 2 + 1)

        # masking
        self.generator = Generator(model_name)
        # print(get_n_params(self.vocoder))
        # print(get_n_params(self.generator))

    def get_vocoder(self):
        return self.vocoder

    def get_f_helper(self):
        return self.f_helper

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return sp, mel_orig

    def forward(self, mel_orig):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(mel_orig)


def to_log(input):
    assert torch.sum(input < 0) == 0, (
        str(input) + " has negative values counts " + str(torch.sum(input < 0))
    )
    return torch.log10(torch.clip(input, min=1e-8))


def from_log(input):
    input = torch.clip(input, min=-np.inf, max=5)
    return 10**input


class BN_GRU(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer=1,
        bidirectional=False,
        batchnorm=True,
        dropout=0.0,
    ):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def forward(self, inputs):
        # (batch, 1, seq, feature)
        if self.batchnorm:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)


class Generator(nn.Module):
    def __init__(self, model_name="unet"):
        super(Generator, self).__init__()
        if model_name == "unet":

            self.analysis_module = UNetResComplex_100Mb(channels=1)
        else:
            raise

    def forward(self, mel_orig):
        out = self.analysis_module(to_log(mel_orig))
        if isinstance(out, type({})):
            out = out["mel"]
        mel = out + to_log(mel_orig)
        return {"mel": mel}
