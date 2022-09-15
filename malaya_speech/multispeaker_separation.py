from malaya_speech.supervised import separation
from herpetologist import check_type
from malaya_speech.utils import describe_availability
import logging

logger = logging.getLogger(__name__)

_availability = {
    'fastsep-2': {
        'Size (MB)': 78.7,
        'Quantized Size (MB)': 20.5,
        'SISNR PIT': 14.156882,
    },
    'fastsep-4': {
        'Size (MB)': 155,
        'Quantized Size (MB)': 40.2,
        'SISNR PIT': 19.6825,
    },
    'fastsep-6': {
        'Size (MB)': 231,
        'Quantized Size (MB)': 60,
        'SISNR PIT': 15.0647,
    },
}


def available_deep_wav():
    """
    List available FastSep models trained on raw 8k wav.
    """
    logger.info('Tested on 1k samples')

    return describe_availability(_availability)


@check_type
def deep_wav(model: str = 'fastsep-4', quantized: bool = False, **kwargs):
    """
    Load FastSep model, trained on raw 8k wav using SISNR PIT loss.

    Parameters
    ----------
    model : str, optional (default='fastsep-4')
        Check available models at `malaya_speech.multispeaker_separation.available_deep_wav()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.Split class
    """

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.multispeaker_separation.available_deep_wav()`.'
        )
    return separation.load(
        model=model,
        module='multispeaker-separation-wav',
        quantized=quantized,
        **kwargs
    )
