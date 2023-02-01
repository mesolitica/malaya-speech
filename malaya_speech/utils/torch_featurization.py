"""
https://github.com/pytorch/audio/blob/main/examples/asr/librispeech_conformer_rnnt/transforms.py
"""

import torchaudio
import torch
import math
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)

torchaudio_available = False
try:
    import torchaudio
    torchaudio_available = True
except ModuleNotFoundError:
    logger.warning('`torchaudio` is not available, `malaya_speech.utils.torch_featurization` is not able to use.')

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

if torchaudio_available:
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)
else:
    spectrogram_transform = None


def extract_labels(sp_model, samples):
    targets = [sp_model.encode(sample[2].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


def melspectrogram(x):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    if spectrogram_transform is None:
        raise ModuleNotFoundError(
            'torchaudio not installed. Please install it by `pip install torchaudio` and try again.'
        )
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


class STT_SPM(torch.nn.Module):
    def __init__(self, global_stats_path, sp_model_path):
        try:
            import sentencepiece as spm
        except:
            raise ModuleNotFoundError(
                'sentencepiece not installed. Please install it by `pip install sentencepiece` and try again.'
            )

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.global_stats = GlobalStatsNormalization(global_stats_path)

    def encode(self, samples):
        """
        samples must be [(float32 array, text), (float32 array, text)]
        """
        mel_features = [spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)

        targets = [sp_model.encode(sample[1].lower()) for sample in samples]
        lengths_targets = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)

        return features, lengths, targets, lengths_targets

    def decode(self, samples):
        """
        samples must be [torchaudio.models.Hypothesis, torchaudio.models.Hypothesis]
        """
        tokens_idx = 0
        score_idx = 3
        post_process_remove_list = [
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        ]
        filtered_hypo_tokens = [
            [token_index for token_index in h[tokens_idx][1:] if token_index not in post_process_remove_list] for h in hypos
        ]
        hypos_str = [self.sp_model.decode(s) for s in filtered_hypo_tokens]
        hypos_ids = [h[tokens_idx][1:] for h in hypos]
        hypos_score = [[math.exp(h[score_idx])] for h in hypos]

        nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))
        return nbest_batch
