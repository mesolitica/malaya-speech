from malaya_speech.supervised import classification
from malaya_speech.utils import describe_availability
import logging

logger = logging.getLogger(__name__)

_nemo_availability = {
    'huseinzol05/nemo-is-clean-speakernet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet',
        'Size (MB)': 0.481,
        'macro precision': 0.60044,
        'macro recall': 0.72819,
        'macro f1-score': 0.49225,
    },
    'huseinzol05/nemo-is-clean-titanet_large': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet',
        'Size (MB)': 0.481,
        'macro precision': 0.47633,
        'macro recall': 0.49895,
        'macro f1-score': 0.46915,
    },
}


def available_nemo():
    """
    List available Nvidia Nemo is clean models.
    """

    return describe_availability(_nemo_availability)


def nemo(
    model: str = 'huseinzol05/nemo-is-clean-speakernet',
    **kwargs,
):
    """
    Load Nvidia Nemo is clean model.
    Trained on 100, 200, 300 ms frames.

    Parameters
    ----------
    model : str, optional (default='huseinzol05/nemo-is-clean-speakernet')
        Check available models at `malaya_speech.is_clean.available_nemo()`.

    Returns
    -------
    result : malaya_speech.torch_model.nemo.Classification class
    """
    if model not in _nemo_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.is_clean.available_nemo()`.'
        )

    return classification.nemo_classification(
        model=model,
        label=[False, True],
        **kwargs
    )
