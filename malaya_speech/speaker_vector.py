from malaya_speech.supervised import classification
import logging
import warnings

logger = logging.getLogger(__name__)

# EER calculation, https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/speaker-embedding/calculate-EER
# EER tested on VoxCeleb2 test set.


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

dvector_accuracy = {
    'original from': 'https://github.com/yistLin/dvector',
    'Size (MB)': 5.45,
    'Embedding Size': 256,
    'EER': 0.1356490598298,
}

available_nemo = {
    'huseinzol05/nemo-ecapa-tdnn': {
        'Size (MB)': 96.8,
        'Embedding Size': 192,
        'EER': 0.0249200000000007,
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ecapa_tdnn',
    },
    'huseinzol05/nemo-speakernet': {
        'Size (MB)': 23.6,
        'Embedding Size': 192,
        'EER': 0.0427898305,
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speakerverification_speakernet',
    },
    'huseinzol05/nemo-titanet_large': {
        'Size (MB)': 101.6,
        'Embedding Size': 192,
        'EER': 0.02277999999996,
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large',
    }
}

huggingface_accuracy = {
    'microsoft/wavlm-base-sv': {
        'Size (MB)': 405,
        'Embedding Size': 512,
        'EER': 0.07827375115,
    },
    'microsoft/wavlm-base-plus-sv': {
        'Size (MB)': 405,
        'Embedding Size': 512,
        'EER': 0.06688427572,
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

info = """
tested on VoxCeleb2 test set. Lower EER is better.
download the test set at https://github.com/huseinzol05/malaya-speech/tree/master/data/voxceleb
""".strip()


def nemo(
    model: str = 'huseinzol05/nemo-ecapa-tdnn',
    **kwargs,
):
    """
    Load Nemo Speaker verification model.

    Parameters
    ----------
    model : str, optional (default='huseinzol05/nemo-ecapa-tdnn')
        Check available models at `malaya_speech.speaker_vector.available_nemo`.

    Returns
    -------
    result : malaya_speech.torch_model.nemo.SpeakerVector class
    """
    if model not in available_nemo:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.speaker_vector.available_nemo`.'
        )

    return classification.nemo_speaker_vector(
        model=model,
        **kwargs
    )
