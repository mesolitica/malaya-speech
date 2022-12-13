import torch
import logging

logger = logging.getLogger(__name__)

torchaudio_available = False
try:
    import torchaudio

    torchaudio_available = True
except Exception as e:
    logger.warning(
        'torchaudio is not installed, please install it by `pip install torchaudio` or else not able to use hifigan-bwe')

SAMPLE_RATE = 48000


class BandwidthExtender(torch.nn.Module):
    """HiFi-GAN+ generator model"""

    def __init__(self) -> None:
        super().__init__()

        # store the training sample rate in the state dict, so that
        # we can run inference on a model trained for a different rate
        self.sample_rate: torch.Tensor
        self.register_buffer("sample_rate", torch.tensor(SAMPLE_RATE))

        self._wavenet = WaveNet(
            stacks=2,
            layers=8,
            in_channels=1,
            wavenet_channels=128,
            out_channels=1,
            kernel_size=3,
            dilation_base=3,
        )

    def forward(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # allow simple synthesis over vectors by automatically unsqueezing
        squeeze = len(x.shape) == 1
        if squeeze:
            x = x.unsqueeze(0).unsqueeze(0)

        # first upsample the signal to the target sample rate
        # using bandlimited interpolation
        x = torchaudio.functional.resample(
            x,
            sample_rate,
            self.sample_rate,
            resampling_method="kaiser_window",
            lowpass_filter_width=16,
            rolloff=0.945,
            beta=14.769656459379492,
        )

        # in order to reduce edge artificacts due to residual conv padding,
        # pad the signal with silence before applying the wavenet, then
        # remove the padding afterward to get the desired signal length
        pad = self._wavenet.receptive_field // 2
        x = torch.nn.functional.pad(x, [pad, pad])
        x = torch.tanh(self._wavenet(x))
        x = x[..., pad:-pad]

        # if a single vector was requested, squeeze back to it
        if squeeze:
            x = x.squeeze(0).squeeze(0)

        return x

    def apply_weightnorm(self) -> None:
        self.apply(lambda m: self._apply_conv(torch.nn.utils.weight_norm, m))

    def remove_weightnorm(self) -> None:
        self.apply(lambda m: self._apply_conv(torch.nn.utils.remove_weight_norm, m))

    def _apply_conv(self, fn, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Conv1d):
            fn(module)


class WaveNet(torch.nn.Module):
    """stacked gated residual 1D convolutions
    This is a non-causal, non-conditional variant of the WaveNet architecture
    from van den Oord, et al.
    https://arxiv.org/abs/1609.03499
    """

    def __init__(
        self,
        stacks: int,
        layers: int,
        in_channels: int,
        wavenet_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation_base: int,
    ):
        super().__init__()

        # initial 1x1 convolution to match the residual channels
        self._conv_in = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=wavenet_channels,
            kernel_size=1,
        )

        # stacked gated residual convolution layers
        self._layers = torch.nn.ModuleList()
        for _ in range(stacks):
            for i in range(layers):
                layer = WaveNetLayer(
                    channels=wavenet_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**i,
                )
                self._layers.append(layer)

        # output 1x1 convolution to project to the desired output dimension
        self._conv_out = torch.nn.Conv1d(
            in_channels=wavenet_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        # calculate the network's effective receptive field
        self.receptive_field = (
            (kernel_size - 1) * stacks * sum(dilation_base**i for i in range(layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply the input projection to wavenet channels
        x = self._conv_in(x)

        # apply the wavenet layers
        s = 0
        for n in self._layers:
            x, h = n(x)
            s += h
        x = s * torch.tensor(1.0 / len(self._layers)).sqrt()

        # apply the output projection
        x = self._conv_out(x)

        return x


class WaveNetLayer(torch.nn.Module):
    """a single gated residual wavenet layer"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()

        # combined gate+activation convolution
        self._conv = torch.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )

        # skip connection projection
        self._conv_skip = torch.nn.Conv1d(
            in_channels=channels // 2,
            out_channels=channels,
            kernel_size=1,
        )

        # output projection
        self._conv_out = torch.nn.Conv1d(
            in_channels=channels // 2,
            out_channels=channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        # save off the residual connection
        r = x

        # apply dilated convolution
        x = self._conv(x)

        # split and gate
        x, g = x.split(x.size(1) // 2, dim=1)
        x = torch.tanh(x) * torch.sigmoid(g)

        # apply skip and output convolutions
        s = self._conv_skip(x)
        x = self._conv_out(x)

        # add residual and apply a normalizing gain
        x = (x + r) * torch.tensor(0.5).sqrt()

        return x, s
