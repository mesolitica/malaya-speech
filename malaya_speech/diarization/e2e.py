
from herpetologist import check_type
from malaya_speech.utils import describe_availability
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'microsoft/wavlm-base-sd': {
        'Size (MB)': 378,
    },
    'microsoft/wavlm-base-plus-sd': {
        'Size (MB)': 378,
    },
    'microsoft/unispeech-sat-large-sd': {
        'Size (MB)': 378,
    },
    'microsoft/unispeech-sat-base-plus-sd': {
        'Size (MB)': 378,
    },
    'microsoft/unispeech-sat-base-sd': {
        'Size (MB)': 378,
    },
}


def available_huggingface():
    """
    List available HuggingFace Speaker Diarization models.
    """

    return describe_availability(_huggingface_availability)


@check_type
def huggingface(
    model: str = 'microsoft/wavlm-base-plus-sd',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='microsoft/wavlm-base-plus-sd')
        Check available models at `malaya_speech.force_alignment.seq2seq.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.model.huggingface.FrameClassification class
    """
