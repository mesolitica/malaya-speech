import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import Optional
from torch import Tensor
from torch.nn import Dropout, Conv1d, MultiheadAttention
import numpy as np


class Convertor(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, dropout=0.1):
        super(Convertor, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.conv0 = weight_norm(Conv1d(d_model, d_hid, 5, padding=2))
        self.conv1 = weight_norm(Conv1d(d_hid, d_model, 1, padding=0))
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.conv0.apply(init_weights)
        self.conv1.apply(init_weights)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # len,B,dim
        src2 = src.transpose(0, 1).transpose(1, 2)  # B,dim,len
        src2 = self.conv1(F.relu(self.conv0(src2)))  # B,dim, len
        src2 = src2.transpose(1, 0).transpose(2, 0)
        src = src + self.dropout2(src2)  # len,B,dim
        return src

    def remove_weight_norm(self):
        remove_weight_norm(self.conv0)
        remove_weight_norm(self.conv1)


class Conv1D_Norm_Act(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, act_fn=None):
        super(Conv1D_Norm_Act, self).__init__()
        self.act_fn = act_fn
        self.conv_block = nn.ModuleList()
        self.conv_block.add_module(
            "conv0",
            weight_norm(
                nn.Conv1d(
                    c_in,
                    c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)))
        self.conv_block.apply(init_weights)

    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        if self.act_fn is not None:
            x = F.relu(x)
        return x

    def remove_weight_norm(self):
        for l in self.conv_block:
            remove_weight_norm(l)


class Block_Unit(nn.Module):
    def __init__(self, c_in, c_out, act_fn=None):
        super(Block_Unit, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.conv_block1 = Conv1D_Norm_Act(c_in, c_out, 5, 1, 2, act_fn)
        self.conv_block2 = Conv1D_Norm_Act(c_in, c_out, 3, 1, 1, act_fn)

        self.adjust_dim_layer = weight_norm(
            nn.Conv1d(
                c_in,
                c_out,
                kernel_size=1,
                stride=1,
                padding=0))

        self.adjust_dim_layer.apply(init_weights)

    def forward(self, x):
        out1 = self.conv_block1(x) + self.conv_block2(x)
        if self.c_in != self.c_out:
            x = self.adjust_dim_layer(x)
        x = out1 + x
        return x

    def remove_weight_norm(self):
        self.conv_block1.remove_weight_norm()
        self.conv_block2.remove_weight_norm()
        remove_weight_norm(self.adjust_dim_layer)


class Attention(nn.Module):
    def __init__(self, d_model, d_head):
        super(Attention, self).__init__()
        self.cross_attn = MultiheadAttention(d_model, d_head)

    def forward(self, x):   # [B,dim,len]
        x = x.transpose(1, 2).transpose(1, 0)
        x = self.cross_attn(x, x, x)[0]  # len, B , dim
        x = x.transpose(1, 0).transpose(1, 2)  # B,dim,len
        return x


class Encoder(nn.Module):
    def __init__(self, d_model=192):
        super(Encoder, self).__init__()
        self.pre_block = nn.Sequential(
            Block_Unit(80, 192, nn.ReLU()),
            Block_Unit(192, 128, nn.ReLU()),
            Block_Unit(128, 36, None),

        )
        self.post_block = nn.Sequential(
            Attention(36, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(36)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pre_block(x)  # [B,512,len]
        x = self.post_block(x)  # [B,512,len]
        return x

    def remove_weight_norm(self):
        for l in self.pre_block:
            l.remove_weight_norm()


class Decoder(nn.Module):
    def __init__(self, d_model=192):
        super(Decoder, self).__init__()
        self.pre_conv_block = nn.Sequential(
            Block_Unit(36, d_model, None),
        )
        self.pre_attention_block = nn.Sequential(
            Attention(d_model, 8),
            nn.ReLU(),
            nn.InstanceNorm1d(d_model)
        )

        self.mel_linear1 = nn.Linear(d_model, d_model)
        self.mel_linear2 = nn.Linear(80, 80)

        self.smoothers = nn.Sequential(
            Convertor(d_model, 8, 512),
            Convertor(d_model, 8, 512),
            Convertor(d_model, 8, 512))  # len,B,dim

        self.post_block = nn.Sequential(
            Block_Unit(d_model, d_model, nn.ReLU()),
            Block_Unit(d_model, d_model//2, nn.ReLU()),  # /2
            Block_Unit(d_model//2, 80, nn.ReLU()),  # /2
            Block_Unit(80, 80, nn.ReLU()),  # /2
        )

    def forward(self, x, x_masks):
        x = self.pre_conv_block(x)
        x = self.pre_attention_block(x)
        x = x.transpose(1, 2)  # B,len,dim
        x = self.mel_linear1(x)
        # x = self.mel_linear2(x)  # b ,len ,dim

        x = x.transpose(0, 1)  # [len,B,512]

        for layer in self.smoothers:
            x = layer(x, src_key_padding_mask=x_masks)

        # x = self.smoothers(x, src_key_padding_mask=x_masks)  # [len,B,512]
        # x = self.mel_linear(x)        # #[len,B,512] --->[len,B,80]
        x = x.transpose(0, 1).transpose(1, 2)  # b ,dim ,len
        x = self.post_block(x)
        # x = self.post_net(x)# [B,80,len]
        x = x.transpose(1, 2)  # [B,len,80]
        x = F.relu(self.mel_linear2(x))
        return x

    def remove_weight_norm(self):
        for l in self.pre_conv_block:
            l.remove_weight_norm()
        for l in self.post_block:
            l.remove_weight_norm()
        for l in self.smoothers:
            l.remove_weight_norm()


class Generator(nn.Module):
    def __init__(self, d_model=192):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, x_masks):
        enc = self.encoder(x)
        out = self.decoder(enc, x_masks)
        return out

    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()
        self.decoder.remove_weight_norm()


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)
