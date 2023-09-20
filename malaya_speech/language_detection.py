from malaya_speech.supervised import classification
from malaya_speech.utils import describe_availability
import logging
import warnings

logger = logging.getLogger(__name__)

_availability = {
    'vggvox-v2': {
        'Size (MB)': 30.9,
        'Quantized Size (MB)': 7.92,
        'Accuracy': 0.90204,
    },
    'deep-speaker': {
        'Size (MB)': 96.9,
        'Quantized Size (MB)': 24.4,
        'Accuracy': 0.8945,
    },
}

_nemo_availability = {
    'huseinzol05/nemo-speaker-count-speakernet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet',
        'Size (MB)': 16.2,
    },
    'huseinzol05/nemo-speaker-count-titanet_large': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet',
        'Size (MB)': 88.8,
    },
}

labels = [
    'english',
    'indonesian',
    'malay',
    'mandarin',
    'manglish',
    'others',
    'not a language',
]


def available_model():
    """
    List available language detection deep models.
    """
    warnings.warn(
        '`malaya_speech.language_detection.deep_model` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)
    logger.info('last accuracy during training session before early stopping.')

    return describe_availability(_availability)


def available_nemo():
    """
    List available Nvidia Nemo language detection models.
    """

    return describe_availability(_nemo_availability)


def deep_model(model: str = 'vggvox-v2', quantized: bool = False, **kwargs):
    """
    Load language detection deep model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Check available models at `malaya_speech.language_detection.available_model()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """
    warnings.warn(
        '`malaya_speech.language_detection.deep_model` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.language_detection.available_model()`.'
        )

    settings = {
        'vggvox-v1': {},
        'vggvox-v2': {'concat': False},
        'deep-speaker': {'voice_only': False},
    }

    return classification.load(
        model=model,
        module='language-detection',
        extra=settings[model],
        label=labels,
        quantized=quantized,
        **kwargs
    )


def nemo(
    model: str = 'huseinzol05/nemo-language-detection-speakernet',
    **kwargs,
):
    """
    Load Nvidia Nemo speaker count model.
    Trained on 200, 300 ms frames.

    Parameters
    ----------
    model : str, optional (default='huseinzol05/nemo-language-detection-speakernet')
        Check available models at `malaya_speech.language_detection.available_nemo()`.

    Returns
    -------
    result : malaya_speech.torch_model.nemo.Classification class
    """
    if model not in _nemo_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.language_detection.available_nemo()`.'
        )

    return classification.nemo_classification(
        model=model,
        label=labels,
        **kwargs
    )
