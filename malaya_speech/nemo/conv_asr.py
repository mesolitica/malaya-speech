import torch
import torch.nn as nn
import torch.nn.functional as F
from malaya_speech.nemo.jasper import (
    JasperBlock,
    MaskedConv1d,
    ParallelBlock,
    SqueezeExcite,
    init_weights,
    jasper_activations,
)
from malaya_speech.nemo.tdnn_attention import (
    AttentivePoolLayer,
    StatsPoolLayer,
    TDNNModule,
    TDNNSEModule,
)
from typing import Optional, Union


class ConvASREncoder(torch.nn.Module):
    """
    Convolutional encoder for ASR models. With this class you can implement JasperNet and QuartzNet models.
    Based on these papers:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
    """

    def input_example(self, max_batch=1, max_dim=8192):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        device = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=device)
        lens = torch.full(size=(input_example.shape[0],), fill_value=max_dim, device=device)
        return tuple([input_example, lens])

    def __init__(
        self,
        jasper,
        activation: str,
        feat_in: int,
        normalization_mode: str = "batch",
        residual_mode: str = "add",
        norm_groups: int = -1,
        conv_mask: bool = True,
        frame_splicing: int = 1,
        init_mode: Optional[str] = 'xavier_uniform',

    ):
        super().__init__()

        activation = jasper_activations[activation]()

        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        feat_in = feat_in * frame_splicing

        self._feat_in = feat_in

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            residual_mode = lcfg.get('residual_mode', residual_mode)
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 8)
            se_context_window = lcfg.get('se_context_size', -1)
            se_interpolation_mode = lcfg.get('se_interpolation_mode', 'nearest')
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            stride_last = lcfg.get('stride_last', False)
            future_context = lcfg.get('future_context', -1)
            encoder_layers.append(
                JasperBlock(
                    feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'],
                    residual=lcfg['residual'],
                    groups=groups,
                    separable=separable,
                    heads=heads,
                    residual_mode=residual_mode,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation,
                    residual_panes=dense_res,
                    conv_mask=conv_mask,
                    se=se,
                    se_reduction_ratio=se_reduction_ratio,
                    se_context_window=se_context_window,
                    se_interpolation_mode=se_interpolation_mode,
                    kernel_size_factor=kernel_size_factor,
                    stride_last=stride_last,
                    future_context=future_context,
                )
            )
            feat_in = lcfg['filters']

        self._feat_out = feat_in

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

        self.max_audio_length = 0

    def forward(self, audio_signal, length):
        self.update_max_sequence_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]

        return s_input[-1], length

    def update_max_sequence_length(self, seq_length: int, device):

        if seq_length > self.max_audio_length:
            if seq_length < 5000:
                seq_length = seq_length * 2
            elif seq_length < 10000:
                seq_length = seq_length * 1.5
            self.max_audio_length = seq_length

            device = next(self.parameters()).device
            seq_range = torch.arange(0, self.max_audio_length, device=device)
            if hasattr(self, 'seq_range'):
                self.seq_range = seq_range
            else:
                self.register_buffer('seq_range', seq_range, persistent=False)

            # Update all submodules
            for name, m in self.named_modules():
                if isinstance(m, MaskedConv1d):
                    m.update_masked_length(self.max_audio_length, seq_range=self.seq_range)
                elif isinstance(m, SqueezeExcite):
                    m.set_max_len(self.max_audio_length, seq_range=self.seq_range)


class SpeakerDecoder(torch.nn.Module):
    """
    Speaker Decoder creates the final neural layers that maps from the outputs
    of Jasper Encoder to the embedding layer followed by speaker based softmax loss.
    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of unique speakers in dataset
        emb_sizes (list) : shapes of intermediate embedding layers (we consider speaker embbeddings from 1st of this layers)
                Defaults to [1024,1024]
        pool_mode (str) : Pooling strategy type. options are 'xvector','tap', 'attention'
                Defaults to 'xvector (mean and variance)'
                tap (temporal average pooling: just mean)
                attention (attention based pooling)
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        emb_sizes: Optional[Union[int, list]] = 256,
        pool_mode: str = 'xvector',
        angular: bool = False,
        attention_channels: int = 128,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()
        self.angular = angular
        self.emb_id = 2
        bias = False if self.angular else True
        emb_sizes = [emb_sizes] if type(emb_sizes) is int else emb_sizes

        self._num_classes = num_classes
        self.pool_mode = pool_mode.lower()
        if self.pool_mode == 'xvector' or self.pool_mode == 'tap':
            self._pooling = StatsPoolLayer(feat_in=feat_in, pool_mode=self.pool_mode)
            affine_type = 'linear'
        elif self.pool_mode == 'attention':
            self._pooling = AttentivePoolLayer(inp_filters=feat_in, attention_channels=attention_channels)
            affine_type = 'conv'

        shapes = [self._pooling.feat_in]
        for size in emb_sizes:
            shapes.append(int(size))

        emb_layers = []
        for shape_in, shape_out in zip(shapes[:-1], shapes[1:]):
            layer = self.affine_layer(shape_in, shape_out, learn_mean=False, affine_type=affine_type)
            emb_layers.append(layer)

        self.emb_layers = nn.ModuleList(emb_layers)

        self.final = nn.Linear(shapes[-1], self._num_classes, bias=bias)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def affine_layer(
        self, inp_shape, out_shape, learn_mean=True, affine_type='conv',
    ):
        if affine_type == 'conv':
            layer = nn.Sequential(
                nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True),
                nn.Conv1d(inp_shape, out_shape, kernel_size=1),
            )

        else:
            layer = nn.Sequential(
                nn.Linear(inp_shape, out_shape),
                nn.BatchNorm1d(out_shape, affine=learn_mean, track_running_stats=True),
                nn.ReLU(),
            )

        return layer

    def forward(self, encoder_output, length=None):
        pool = self._pooling(encoder_output, length)
        embs = []

        for layer in self.emb_layers:
            pool, emb = layer(pool), layer[: self.emb_id](pool)
            embs.append(emb)

        pool = pool.squeeze(-1)
        if self.angular:
            for W in self.final.parameters():
                W = F.normalize(W, p=2, dim=1)
            pool = F.normalize(pool, p=2, dim=1)

        out = self.final(pool)

        return out, embs[-1].squeeze(-1)


class ECAPAEncoder(torch.nn.Module):
    """
    Modified ECAPA Encoder layer without Res2Net module for faster training and inference which achieves
    better numbers on speaker diarization tasks
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    input:
        feat_in: input feature shape (mel spec feature shape)
        filters: list of filter shapes for SE_TDNN modules
        kernel_sizes: list of kernel shapes for SE_TDNN modules
        dilations: list of dilations for group conv se layer
        scale: scale value to group wider conv channels (deafult:8)
    output:
        outputs : encoded output
        output_length: masked output lengths
    """

    def __init__(
        self,
        feat_in: int,
        filters: list,
        kernel_sizes: list,
        dilations: list,
        scale: int = 8,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TDNNModule(feat_in, filters[0], kernel_size=kernel_sizes[0], dilation=dilations[0]))

        for i in range(len(filters) - 2):
            self.layers.append(
                TDNNSEModule(
                    filters[i],
                    filters[i + 1],
                    group_scale=scale,
                    se_channels=128,
                    kernel_size=kernel_sizes[i + 1],
                    dilation=dilations[i + 1],
                )
            )
        self.feature_agg = TDNNModule(filters[-1], filters[-1], kernel_sizes[-1], dilations[-1])
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length=None):
        x = audio_signal
        outputs = []

        for layer in self.layers:
            x = layer(x, length=length)
            outputs.append(x)

        x = torch.cat(outputs[1:], dim=1)
        x = self.feature_agg(x)
        return x, length


class ConvASRDecoderClassification(torch.nn.Module):
    """Simple ASR Decoder for use with classification models such as JasperNet and QuartzNet
     Based on these papers:
        https://arxiv.org/pdf/2005.04290.pdf
    """

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        init_mode: Optional[str] = "xavier_uniform",
        return_logits: bool = True,
        pooling_type='avg',
    ):
        super().__init__()

        self._feat_in = feat_in
        self._return_logits = return_logits
        self._num_classes = num_classes

        if pooling_type == 'avg':
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max':
            self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError('Pooling type chosen is not valid. Must be either `avg` or `max`')

        self.decoder_layers = torch.nn.Sequential(torch.nn.Linear(self._feat_in, self._num_classes, bias=True))
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        batch, in_channels, timesteps = encoder_output.size()

        encoder_output = self.pooling(encoder_output).view(batch, in_channels)  # [B, C]
        logits = self.decoder_layers(encoder_output)  # [B, num_classes]

        if self._return_logits:
            return logits

        return torch.nn.functional.softmax(logits, dim=-1)

    @property
    def num_classes(self):
        return self._num_classes
