# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

version = '1.1'
bump_version = '1.1'
__version__ = bump_version

import malaya_boilerplate

malaya_boilerplate.__package__ = 'malaya-speech'
malaya_boilerplate.__url__ = (
    'https://f000.backblazeb2.com/file/malaya-speech-model/'
)
malaya_boilerplate.__package_version__ = version

from malaya_boilerplate.utils import get_home

__home__, _ = get_home()

from . import augmentation
from . import config
from . import extra
from . import utils
from . import age_detection
from . import diarization
from . import emotion
from . import gender
from . import language_detection
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
from .streaming import *
from .utils import *
