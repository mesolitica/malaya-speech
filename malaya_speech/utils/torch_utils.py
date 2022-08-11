import torch
import numpy as np


def from_log(input):
    input = torch.clip(input, min=-np.inf, max=5)
    return 10**input


def to_tensor_cuda(tensor, cuda):
    if cuda and torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor.cpu()


def to_tensor(array):
    if 'float' in str(array.dtype):
        x = torch.Tensor(array)
    elif 'int' in str(array.dtype):
        x = torch.LongTensor(array)
    else:
        return x
    return x


def to_numpy(tensor):
    if 'cuda' in str(tensor.device):
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()
