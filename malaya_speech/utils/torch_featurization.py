"""
https://github.com/pytorch/audio/blob/main/examples/asr/librispeech_conformer_rnnt/transforms.py
"""

import torch
import torchaudio
import math
import logging
import numpy as np
import json
from torchaudio.transforms import Fade
from torchaudio.models import Hypothesis, RNNTBeamSearch
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

try:
    from torchaudio.io import StreamReader
except Exception as e:
    logger.warning(f'torchaudio.io.StreamReader exception: {e}')
    logger.warning(
        '`torchaudio.io.StreamReader` is not available, `malaya_speech.streaming.torchaudio.stream` is not able to use.')
    StreamReader = None

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)


def post_process_hypos(
    hypos: List[Hypothesis], sp_model
) -> List[Tuple[str, float, List[int], List[int]]]:
    tokens_idx = 0
    score_idx = 3
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    filtered_hypo_tokens = [
        [token_index for token_index in h[tokens_idx]
         [1:] if token_index not in post_process_remove_list] for h in hypos
    ]
    hypos_str = [sp_model.decode(s) for s in filtered_hypo_tokens]
    hypos_ids = [h[tokens_idx][1:] for h in hypos]
    hypos_score = [[math.exp(h[score_idx])] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))

    return nbest_batch


def melspectrogram(x):
    if spectrogram_transform is None:
        raise ModuleNotFoundError(
            'torchaudio not installed. Please install it by `pip install torchaudio` and try again.'
        )
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    return spectrogram_transform(x).transpose(1, 0)


def piecewise_linear_log(x):
    x = x * GAIN
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob['mean'])
        self.invstddev = torch.tensor(blob['invstddev'])

    def forward(self, input):
        return (input - self.mean) * self.invstddev


class FeatureExtractor(torch.nn.Module):
    def __init__(self, global_stats_path, pad=False):
        super().__init__()

        self.global_stats = GlobalStatsNormalization(global_stats_path=global_stats_path)
        self.pad = pad

    def forward(self, input):
        mel = melspectrogram(input)
        mel = piecewise_linear_log(mel)
        mel = self.global_stats(mel)
        if self.pad:
            mel = torch.nn.functional.pad(mel, pad=(0, 0, 0, 4))
        return mel, torch.tensor([len(mel)])


def separate_sources(
    model,
    mix,
    segment=10.,
    overlap=0.1,
    device=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """

    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape='linear')

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final
