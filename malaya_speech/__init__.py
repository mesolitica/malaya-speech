# Malaya-Speech, Speech-Toolkit library for bahasa Malaysia
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya-speech.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/malaya-speech/blob/master/LICENSE

from malaya_boilerplate.utils import get_home

version = '1.4'
bump_version = '1.4.0'
__version__ = bump_version

package = 'malaya-speech'
url = 'https://f000.backblazeb2.com/file/malaya-speech-model/'
__home__, _ = get_home(package=package, package_version=version)

from . import augmentation
from . import config
from . import extra
from . import utils
from . import age_detection
from . import diarization
from . import emotion
from . import force_alignment
from . import gender
from . import is_clean
from . import language_detection
from . import language_model
from . import multispeaker_separation
from . import noise_reduction
from . import speaker_change
from . import speaker_overlap
from . import speaker_vector
from . import speech_enhancement
from . import speechsplit_conversion
from . import stack
from . import stt
from . import super_resolution
from . import tts
from . import vad
from . import vocoder
from . import voice_conversion
from . import utils

from .pipeline import Pipeline
from . import streaming
from .utils import (
    aligner,
    arange,
    astype,
    char,
    combine,
    featurization,
    generator,
    group,
    metrics,
    padding,
    split,
    subword,
    tf_featurization
)
from .utils.read import load, resample
