# Malaya-Speech, Speech-Toolkit library for bahasa Malaysia
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya-speech.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/malaya-speech/blob/master/LICENSE

# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-google-speech-malay-dataset.ipynb
# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-google-speech-ms-fleurs102.ipynb
# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-google-speech-singlish-dataset.ipynb

google_accuracy = {
    'malay-malaya': {
        'WER': 0.16477548774,
        'CER': 0.05973209121,
    },
    'malay-fleur102': {
        'WER': 0.109588779,
        'CER': 0.047891527,
    },
    'singlish': {
        'WER': 0.4941349,
        'CER': 0.3026296,
    },
}

# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-whisper-tiny.ipynb
# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-whisper-base.ipynb
# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-whisper-small.ipynb
# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-whisper-medium.ipynb
# https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/prepare-stt/benchmark-whisper-large-v2.ipynb

whisper_accuracy = {
    'tiny': {
        'Size (MB)': 72.1,
        'malay-malaya': {
            'WER': 0.7897730947,
            'CER': 0.341671582346,
        },
        'malay-fleur102': {
            'WER': 0.640224185,
            'CER': 0.2869274323,
        },
        'singlish': {
            'WER': 0.4751720563,
            'CER': 0.35132630877,
        },
    },
    'base': {
        'Size (MB)': 139,
        'malay-malaya': {
            'WER': 0.5138481614,
            'CER': 0.19487665487,
        },
        'malay-fleur102': {
            'WER': 0.4268323797,
            'CER': 0.1545261803,
        },
        'singlish': {
            'WER': 0.5354453439,
            'CER': 0.4287910359,
        },
    },
    'small': {
        'Size (MB)': 461,
        'malay-malaya': {
            'WER': 0.2818371132,
            'CER': 0.09588120693,
        },
        'malay-fleur102': {
            'WER': 0.2436472703,
            'CER': 0.09136925680,
        },
        'singlish': {
            'WER': 0.5971608337,
            'CER': 0.5003890601,
        },
    },
    'medium': {
        'Size (MB)': 1400,
        'malay-malaya': {
            'WER': 0.18945585961,
            'CER': 0.0658303076,
        },
        'malay-fleur102': {
            'WER': 0.1647166507,
            'CER': 0.065537127,
        },
        'singlish': {
            'WER': 0.68563087121,
            'CER': 0.601676254253,
        },
    },
    'large-v2': {
        'Size (MB)': 2900,
        'malay-malaya': {
            'WER': 0.1585939185,
            'CER': 0.054978161091,
        },
        'malay-fleur102': {
            'WER': 0.127483122485,
            'CER': 0.05648688907,
        },
        'singlish': {
            'WER': 0.6174993839,
            'CER': 0.54582068858,
        },
    }
}

info = """
for `malay-fleur102` language, tested on FLEURS102 `ms_my` test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt
for `malay-malaya` language, tested on malaya-speech test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt
for `singlish` language, tested on IMDA malaya-speech test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt
for `whisper-mixed` language, tested on semisupervised Whisper Large V2 test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt
Malaysian STT Leaderboard at https://huggingface.co/spaces/mesolitica/malaysian-stt-leaderboard
""".strip()

from . import ctc
from . import seq2seq
from . import transducer
