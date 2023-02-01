# Malaya-Speech, Speech-Toolkit library for bahasa Malaysia
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya-speech.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/malaya-speech/blob/master/LICENSE

import logging

logger = logging.getLogger(__name__)

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


def _describe():
    logger.info('for `malay-fleur102` language, tested on FLEURS102 `ms_my` test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt')
    logger.info('for `malay-malaya` language, tested on malaya-speech test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt')
    logger.info('for `singlish` language, tested on IMDA malaya-speech test set, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt')


from . import ctc
from . import seq2seq
from . import transducer
