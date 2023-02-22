# import pytorch_lightning as pl

import torch.utils
from malaya_speech.torch_model.voicefixer.tools.mel_scale import MelScale
import torch.utils.data
from malaya_speech.torch_model.voicefixer.vocoder.base import Vocoder
from malaya_speech.torch_model.voicefixer.tools.pytorch_util import *
from malaya_speech.torch_model.voicefixer.restorer.model_kqq_bn import UNetResComplex_100Mb
from malaya_speech.torch_model.voicefixer.tools.random_ import *
from malaya_speech.torch_model.voicefixer.tools.wav import *
from malaya_speech.torch_model.voicefixer.tools.modules.fDomainHelper import FDomainHelper

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
EPS = 1e-8


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
    def __init__(self, n_mel, hidden, channels):
        super(Generator, self).__init__()
        # todo the currently running trail don't have dropout
        self.denoiser = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Linear(n_mel, n_mel * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),
            nn.Linear(n_mel * 2, n_mel * 4),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            BN_GRU(
                input_dim=n_mel * 4,
                hidden_dim=n_mel * 2,
                bidirectional=True,
                layer=2,
                batchnorm=True,
            ),
            BN_GRU(
                input_dim=n_mel * 4,
                hidden_dim=n_mel * 2,
                bidirectional=True,
                layer=2,
                batchnorm=True,
            ),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Linear(n_mel * 4, n_mel * 4),
            nn.Dropout(0.5),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Linear(n_mel * 4, n_mel),
            nn.Sigmoid(),
        )

        self.unet = UNetResComplex_100Mb(channels=channels)

    def forward(self, sp, mel_orig):
        # Denoising
        noisy = mel_orig.clone()
        clean = self.denoiser(noisy) * noisy
        x = to_log(clean.detach())
        unet_in = torch.cat([to_log(mel_orig), x], dim=1)
        # unet_in = lstm_out
        unet_out = self.unet(unet_in)["mel"]
        # masks
        mel = unet_out + x
        # todo mel and addition here are in log scales
        return {
            "mel": mel,
            "lstm_out": unet_out,
            "unet_out": unet_out,
            "noisy": noisy,
            "clean": clean,
        }


class VoiceFixer(nn.Module):
    def __init__(
        self,
        channels,
        vocoder_pth,
        type_target="vocals",
        nsrc=1,
        loss="l1",
        lr=0.002,
        gamma=0.9,
        batchsize=None,
        frame_length=None,
        sample_rate=None,
        warm_up_steps=1000,
        reduce_lr_steps=15000,
        # datas
        check_val_every_n_epoch=5,
    ):
        super(VoiceFixer, self).__init__()

        if sample_rate == 44100:
            window_size = 2048
            hop_size = 441
            n_mel = 128
        elif sample_rate == 24000:
            window_size = 768
            hop_size = 240
            n_mel = 80
        elif sample_rate == 16000:
            window_size = 512
            hop_size = 160
            n_mel = 80
        else:
            raise ValueError(
                "Error: Sample rate " + str(sample_rate) + " not supported"
            )

        center = (True,)
        pad_mode = "reflect"
        window = "hann"
        freeze_parameters = True

        # self.save_hyperparameters()
        self.nsrc = nsrc
        self.type_target = type_target
        self.channels = channels
        self.lr = lr
        self.generated = None
        self.gamma = gamma
        self.sample_rate = sample_rate
        self.sample_rate = sample_rate
        self.batchsize = batchsize
        self.frame_length = frame_length
        # self.hparams['channels'] = 2

        # self.am = AudioMetrics()
        # self.im = ImgMetrics()

        self.vocoder = Vocoder(vocoder_pth, sample_rate=44100)

        self.valid = None
        self.fake = None

        self.train_step = 0
        self.val_step = 0
        self.val_result_save_dir = None
        self.val_result_save_dir_step = None
        self.downsample_ratio = 2**6  # This number equals 2^{#encoder_blcoks}
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.f_helper = FDomainHelper(
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            pad_mode=pad_mode,
            window=window,
            freeze_parameters=freeze_parameters,
        )

        hidden = window_size // 2 + 1

        self.mel = MelScale(n_mels=n_mel, sample_rate=sample_rate, n_stft=hidden)

        # masking
        self.generator = Generator(n_mel, hidden, channels)

        self.lr_lambda = lambda step: self.get_lr_lambda(
            step,
            gamma=self.gamma,
            warm_up_steps=warm_up_steps,
            reduce_lr_steps=reduce_lr_steps,
        )

        self.lr_lambda_2 = lambda step: self.get_lr_lambda(
            step, gamma=self.gamma, warm_up_steps=10, reduce_lr_steps=reduce_lr_steps
        )

        self.mel_weight_44k_128 = (
            torch.tensor(
                [
                    19.40951426,
                    19.94047336,
                    20.4859038,
                    21.04629067,
                    21.62194148,
                    22.21335214,
                    22.8210215,
                    23.44529231,
                    24.08660962,
                    24.74541882,
                    25.42234287,
                    26.11770576,
                    26.83212784,
                    27.56615283,
                    28.32007747,
                    29.0947679,
                    29.89060111,
                    30.70832636,
                    31.54828121,
                    32.41121487,
                    33.29780773,
                    34.20865341,
                    35.14437675,
                    36.1056621,
                    37.09332763,
                    38.10795802,
                    39.15039691,
                    40.22119881,
                    41.32154931,
                    42.45172373,
                    43.61293329,
                    44.80609379,
                    46.031602,
                    47.29070223,
                    48.58427549,
                    49.91327905,
                    51.27863232,
                    52.68119708,
                    54.1222372,
                    55.60274206,
                    57.12364703,
                    58.68617876,
                    60.29148652,
                    61.94081306,
                    63.63501986,
                    65.37562658,
                    67.16408954,
                    69.00109084,
                    70.88850318,
                    72.82736101,
                    74.81985537,
                    76.86654792,
                    78.96885475,
                    81.12900906,
                    83.34840929,
                    85.62810662,
                    87.97005418,
                    90.37689804,
                    92.84887686,
                    95.38872881,
                    97.99777002,
                    100.67862715,
                    103.43232942,
                    106.26140638,
                    109.16827015,
                    112.15470471,
                    115.22184756,
                    118.37439245,
                    121.6122689,
                    124.93877158,
                    128.35661454,
                    131.86761321,
                    135.47417938,
                    139.18059494,
                    142.98713744,
                    146.89771854,
                    150.91684347,
                    155.0446638,
                    159.28614648,
                    163.64270198,
                    168.12035831,
                    172.71749158,
                    177.44220154,
                    182.29556933,
                    187.28286676,
                    192.40502126,
                    197.6682721,
                    203.07516896,
                    208.63088733,
                    214.33770931,
                    220.19910108,
                    226.22363072,
                    232.41087124,
                    238.76803591,
                    245.30079083,
                    252.01064464,
                    258.90261676,
                    265.98474,
                    273.26010248,
                    280.73496362,
                    288.41440094,
                    296.30489752,
                    304.41180337,
                    312.7377183,
                    321.28877878,
                    330.07870237,
                    339.10812951,
                    348.38276173,
                    357.91393924,
                    367.70513992,
                    377.76413924,
                    388.09467408,
                    398.70920178,
                    409.61813793,
                    420.81980127,
                    432.33215467,
                    444.16083117,
                    456.30919947,
                    468.78589276,
                    481.61325588,
                    494.78824596,
                    508.31969844,
                    522.2238331,
                    536.51163441,
                    551.18859414,
                    566.26142988,
                    581.75006061,
                    597.66210737,
                ]
            )
            / 19.40951426
        )
        self.mel_weight_44k_128 = self.mel_weight_44k_128[None, None, None, ...]

        self.g_loss_weight = 0.01
        self.d_loss_weight = 1

    def get_vocoder(self):
        return self.vocoder

    def get_f_helper(self):
        return self.f_helper

    def get_lr_lambda(self, step, gamma, warm_up_steps, reduce_lr_steps):
        r"""Get lr_lambda for LambdaLR. E.g.,

        .. code-block: python
            lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)

            from torch.optim.lr_scheduler import LambdaLR
            LambdaLR(optimizer, lr_lambda)
        """
        if step <= warm_up_steps:
            return step / warm_up_steps
        else:
            return gamma ** (step // reduce_lr_steps)

    def init_weights(self, module: nn.Module):
        for m in module.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return sp, mel_orig

    def forward(self, sp, mel_orig):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(sp, mel_orig)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            [{"params": self.generator.parameters()}],
            lr=self.lr,
            amsgrad=True,
            betas=(0.5, 0.999),
        )
        optimizer_d = torch.optim.Adam(
            [{"params": self.discriminator.parameters()}],
            lr=self.lr,
            amsgrad=True,
            betas=(0.5, 0.999),
        )

        scheduler_g = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer_g, self.lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        scheduler_d = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer_d, self.lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def preprocess(self, batch, train=False, cutoff=None):
        if train:
            vocal = batch[self.type_target]  # final target
            noise = batch["noise_LR"]  # augmented low resolution audio with noise
            augLR = batch[
                self.type_target + "_aug_LR"
            ]  # # augment low resolution audio
            LR = batch[self.type_target + "_LR"]
            # embed()
            vocal, LR, augLR, noise = (
                vocal.float().permute(0, 2, 1),
                LR.float().permute(0, 2, 1),
                augLR.float().permute(0, 2, 1),
                noise.float().permute(0, 2, 1),
            )
            # LR, noise = self.add_random_noise(LR, noise)
            snr, scale = [], []
            for i in range(vocal.size()[0]):
                (
                    vocal[i, ...],
                    LR[i, ...],
                    augLR[i, ...],
                    noise[i, ...],
                    _snr,
                    _scale,
                ) = add_noise_and_scale_with_HQ_with_Aug(
                    vocal[i, ...],
                    LR[i, ...],
                    augLR[i, ...],
                    noise[i, ...],
                    snr_l=-5,
                    snr_h=45,
                    scale_lower=0.6,
                    scale_upper=1.0,
                )
                snr.append(_snr), scale.append(_scale)
            # vocal, LR = self.amp_to_original_f(vocal, LR)
            # noise = (noise * 0.0) + 1e-8 # todo
            return vocal, augLR, LR, noise + augLR
        else:
            if cutoff is None:
                LR_noisy = batch["noisy"]
                LR = batch["vocals"]
                vocals = batch["vocals"]
                vocals, LR, LR_noisy = (
                    vocals.float().permute(0, 2, 1),
                    LR.float().permute(0, 2, 1),
                    LR_noisy.float().permute(0, 2, 1),
                )
                return vocals, LR, LR_noisy, batch["fname"][0]
            else:
                LR_noisy = batch["noisy" + "LR" + "_" + str(cutoff)]
                LR = batch["vocals" + "LR" + "_" + str(cutoff)]
                vocals = batch["vocals"]
                vocals, LR, LR_noisy = (
                    vocals.float().permute(0, 2, 1),
                    LR.float().permute(0, 2, 1),
                    LR_noisy.float().permute(0, 2, 1),
                )
                return vocals, LR, LR_noisy, batch["fname"][0]

    def clip(self, *args):
        val_max, val_min = [], []
        for each in args:
            val_max.append(torch.max(each))
            val_min.append(torch.min(each))
        return max(val_max), min(val_min)