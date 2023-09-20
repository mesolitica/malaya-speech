from malaya_speech.supervised import unet
from malaya_speech.utils.astype import int_to_float
from malaya_speech.utils import describe_availability
import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

# https://github.com/sigsep/sigsep-mus-eval/blob/master/museval/__init__.py#L364
# Only calculate SDR, ISR, SAR on voice sample

_availability = {
    'unet': {
        'Size (MB)': 78.9,
        'Quantized Size (MB)': 20,
        'SUM MAE': 0.862316,
        'MAE_SPEAKER': 0.460676,
        'MAE_NOISE': 0.401640,
        'SDR': 9.17312,
        'ISR': 13.92435,
        'SAR': 13.20592,
        'sample rate': 44100,
        'instrument': ['voice', 'noise'],
    },
    'resnet-unet': {
        'Size (MB)': 96.4,
        'Quantized Size (MB)': 24.6,
        'SUM MAE': 0.82535,
        'MAE_SPEAKER': 0.43885,
        'MAE_NOISE': 0.38649,
        'SDR': 9.45413,
        'ISR': 13.9639,
        'SAR': 13.60276,
        'sample rate': 44100,
        'instrument': ['voice', 'noise'],
    },
    'resnext-unet': {
        'Size (MB)': 75.4,
        'Quantized Size (MB)': 19,
        'SUM MAE': 0.81102,
        'MAE_SPEAKER': 0.44719,
        'MAE_NOISE': 0.363830,
        'SDR': 8.992832,
        'ISR': 13.49194,
        'SAR': 13.13210,
        'sample rate': 44100,
        'instrument': ['voice', 'noise'],
    },
    'resnet-unet-v2': {
        'Size (MB)': 96.4,
        'Quantized Size (MB)': 24.6,
        'SUM MAE': 0.82535,
        'MAE_SPEAKER': 0.43885,
        'MAE_NOISE': 0.38649,
        'SDR': 9.45413,
        'ISR': 13.9639,
        'SAR': 13.60276,
        'sample rate': 44100,
        'instrument': ['voice'],
    },
}


def available_model():
    """
    List available Noise Reduction deep learning models.
    """

    warnings.warn(
        '`malaya_speech.noise_reduction.deep_model` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    logger.info('Only calculate SDR, ISR, SAR on voice sample. Higher is better.')

    return describe_availability(_availability)


def deep_model(model: str = 'resnet-unet', quantized: bool = False, **kwargs):
    """
    Load Noise Reduction deep learning model.

    Parameters
    ----------
    model : str, optional (default='resnet-unet')
        Check available models at `malaya_speech.noise_reduction.available_model()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.UNET_STFT class
    """

    warnings.warn(
        '`malaya_speech.noise_reduction.deep_model` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.noise_reduction.available_model()`.'
        )

    return unet.load_stft(
        model=model,
        module='noise-reduction',
        instruments=['voice', 'noise'],
        quantized=quantized,
        **kwargs
    )


def torchaudio(**kwargs):
    """
    Load TorchAudio Hybrid Demucs, https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html
    """
    return unet.load_torchaudio(**kwargs)
