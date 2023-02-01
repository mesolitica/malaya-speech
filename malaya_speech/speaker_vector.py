from malaya_speech.supervised import classification
from malaya_speech.utils import describe_availability
from herpetologist import check_type
import logging
import warnings

logger = logging.getLogger(__name__)

# EER calculation, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/speaker-embedding/calculate-EER
# EER tested on VoxCeleb2 test set.

_availability = {
    'deep-speaker': {
        'Size (MB)': 96.7,
        'Quantized Size (MB)': 24.4,
        'Embedding Size': 512,
        'EER': 0.2187,
    },
    'vggvox-v1': {
        'Size (MB)': 70.8,
        'Quantized Size (MB)': 17.7,
        'Embedding Size': 1024,
        'EER': 0.13944,
    },
    'vggvox-v2': {
        'Size (MB)': 43.2,
        'Quantized Size (MB)': 7.92,
        'Embedding Size': 512,
        'EER': 0.0446,
    },
    'speakernet': {
        'Size (MB)': 35,
        'Quantized Size (MB)': 8.88,
        'Embedding Size': 7205,
        'EER': 0.3000285,
    },
    'conformer-base': {
        'Size (MB)': 99.4,
        'Quantized Size (MB)': 27.2,
        'Embedding Size': 512,
        'EER': 0.06938,
    },
    'conformer-tiny': {
        'Size (MB)': 20.3,
        'Quantized Size (MB)': 6.21,
        'Embedding Size': 512,
        'EER': 0.08687,
    },
}

trillsson_accuracy = {
    'trillsson-1': {
        'url': 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson1/1',
        'EER': 0.3804599,
    },
    'trillsson-2': {
        'url': 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson2/1',
        'EER': 0.3898799,
    }
}

nemo_accuracy = {
    'ecapa-tdnn': {
        'Size (MB)': 20.3,
        'Embedding Size': 512,
        'EER': 0.08687,
    },
    'speakernet': {
        'Size (MB)': 20.3,
        'Embedding Size': 512,
        'EER': 0.08687,
    },
    'titanet_large': {
        'Size (MB)': 20.3,
        'Embedding Size': 512,
        'EER': 0.08687,
    }
}

_huggingface_availability = {
    'microsoft/wavlm-base-sv': {
        'Size (MB)': 405,
        'Embedding Size': 512,
        'EER': 0.076128169,
    },
    'microsoft/wavlm-base-plus-sv': {
        'Size (MB)': 405,
        'Embedding Size': 512,
        'EER': 0.0631559379,
    },
    'microsoft/unispeech-sat-large-sv': {
        'Size (MB)': 1290,
        'Embedding Size': 512,
        'EER': 0.2032767553,
    },
    'microsoft/unispeech-sat-base-sv': {
        'Size (MB)': 404,
        'Embedding Size': 512,
        'EER': 0.0782815656,
    },
    'microsoft/unispeech-sat-base-plus-sv': {
        'Size (MB)': 404,
        'Embedding Size': 512,
        'EER': 0.0761281698,
    },
}


def _describe():
    logger.info('tested on VoxCeleb2 test set. Lower EER is better.')
    logger.info('download the test set at https://github.com/huseinzol05/malaya-speech/tree/master/data/voxceleb')


def available_model():
    """
    List available speaker vector deep models.
    """

    _describe()
    return describe_availability(_availability)


def available_nemo():
    """
    List available Nvidia Nemo Speaker vector models.
    """

    _describe()
    return describe_availability(nemo_accuracy)


def available_huggingface():
    """
    List available HuggingFace Speaker vector models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


@check_type
def deep_model(model: str = 'vggvox-v2', quantized: bool = False, **kwargs):
    """
    Load Speaker2Vec model.

    Parameters
    ----------
    model : str, optional (default='speakernet')
        Check available models at `malaya_speech.speaker_vector.available_model()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.speaker_vector.available_model()`.'
        )

    return classification.load(
        model=model,
        module='speaker-vector',
        extra={},
        label=None,
        quantized=quantized,
        **kwargs
    )


@check_type
def huggingface(
    model: str = 'microsoft/wavlm-base-plus-sv',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='microsoft/wavlm-base-plus-sv')
        Check available models at `malaya_speech.speaker_vector.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.torch_model.huggingface.XVector class
    """

    return classification.huggingface_xvector(
        model=model,
        **kwargs
    )
