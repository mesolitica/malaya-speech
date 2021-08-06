# https://github.com/kaparoo/Conv-TasNet

from .model import Model


class ConvTasNetParam:

    __slots__ = 'causal', 'That', 'C', 'L', 'N', 'B', 'Sc', 'H', 'P', 'X', 'R', 'overlap'

    def __init__(self,
                 causal: bool = False,
                 That: int = 128,
                 C: int = 4,
                 L: int = 16,
                 N: int = 512,
                 B: int = 128,
                 Sc: int = 128,
                 H: int = 512,
                 P: int = 3,
                 X: int = 8,
                 R: int = 3,
                 overlap: int = 8):

        if overlap * 2 > L:
            raise ValueError('`overlap` cannot be greater than half of `L`!')

        self.causal = causal
        self.That = That
        self.C = C
        self.L = L
        self.N = N
        self.B = B
        self.Sc = Sc
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.overlap = overlap
