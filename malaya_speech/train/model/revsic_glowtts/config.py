"""
MIT License

Copyright (c) 2021 YoungJoong Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class Config:
    """Configuration
    """

    def __init__(self, mel: int, vocabs: int):
        """Initializer.
        Args:
            mel: channels of the mel-spectrogram.
            vocabs: the number of the vocabularies.
        """
        self.mel = mel
        self.factor = 2
        self.neck = mel * self.factor
        self.vocabs = vocabs

        # standard deviation of isotropic gaussian assumption
        self.temperature = 0.333
        self.noise_scale_w = 1.0
        self.length_scale = 1.0

        # model
        self.channels = 192
        # prenet
        self.prenet_kernel = 5
        self.prenet_layers = 3
        self.prenet_groups = 4
        self.prenet_dropout = 0.5

        # encoder
        self.block_num = 6
        self.block_ffn = self.channels * 4
        self.block_heads = 2
        self.block_dropout = 0.1

        # decoder
        self.flow_groups = 4
        self.flow_block_num = 12
        self.wavenet_block_num = 4
        self.wavenet_cycle = 1
        self.wavenet_kernel_size = 5
        self.wavenet_dilation = 1

        # durator model
        self.dur_kernel = 3
        self.dur_layers = 2
        self.dur_dropout = self.block_dropout
