from malaya_speech.path import (
    PATH_SPEECH_ENHANCEMENT,
    S3_PATH_SPEECH_ENHANCEMENT,
)
from malaya_speech.supervised import unet
from herpetologist import check_type


_availability = {
    'unet': {
        'Size (MB)': 97.8,
        'SUM MAE': 0.0003,
        'MAE_SPEAKER': 0,
        'MAE_NOISE': 0,
    },
    'resnet34-unet': {
        'Size (MB)': 97.8,
        'SUM MAE': 0.0003,
        'MAE_SPEAKER': 0,
        'MAE_NOISE': 0,
    },
}
