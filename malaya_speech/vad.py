from malaya_speech.model.webrtc import WebRTC
from malaya_speech.supervised import classification
from malaya_speech.utils import describe_availability
from herpetologist import check_type
import warnings

# https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/vad/evaluate
_availability = {
    'vggvox-v2': {
        'Size (MB)': 31.1,
        'Quantized Size (MB)': 7.92,
        'macro precision': 0.65847,
        'macro recall': 0.82243,
        'macro f1-score': 0.67149,
    },
    'marblenet-factor1': {
        'Size (MB)': 0.526,
        'Quantized Size (MB)': 0.232,
        'macro precision': 0.64789,
        'macro recall': 0.64168,
        'macro f1-score': 0.64467,
    },
    'marblenet-factor3': {
        'Size (MB)': 3.21,
        'Quantized Size (MB)': 0.934,
        'macro precision': 0.57310,
        'macro recall': 0.61392,
        'macro f1-score': 0.58050,
    },
    'marblenet-factor5': {
        'Size (MB)': 8.38,
        'Quantized Size (MB)': 2.21,
        'macro precision': 0.54572,
        'macro recall': 0.58929,
        'macro f1-score': 0.53424,
    }
}

_nemo_availability = {
    'huseinzol05/nemo-vad-marblenet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet',
        'Size (MB)': 0.481,
        'macro precision': 0.60044,
        'macro recall': 0.72819,
        'macro f1-score': 0.49225,
    },
    'huseinzol05/nemo-vad-multilingual-marblenet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet',
        'Size (MB)': 0.481,
        'macro precision': 0.47633,
        'macro recall': 0.49895,
        'macro f1-score': 0.46915,
    },
}

silero_vad = {
    'original from': 'https://github.com/snakers4/silero-vad',
    'macro precision': 0.60044,
    'macro recall': 0.72819,
    'macro f1-score': 0.49225,
}


def available_model():
    """
    List available VAD deep models.
    """

    return describe_availability(_availability)


def available_nemo():
    """
    List available Nvidia Nemo VAD models.
    """

    return describe_availability(_nemo_availability)


@check_type
def webrtc(
    aggressiveness: int = 3,
    sample_rate: int = 16000,
    minimum_amplitude: int = 100,
):
    """
    Load WebRTC VAD model. 
    WebRTC prefer 30ms frame, https://github.com/wiseman/py-webrtcvad#how-to-use-it

    {
        'macro precision': 0.47163,
        'macro recall': 0.46148,
        'macro f1-score': 0.46461,
    }
    https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/vad/evaluate/webrtc.ipynb

    Parameters
    ----------
    aggressiveness: int, optional (default=3)
        an integer between 0 and 3.
        0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
    sample_rate: int, optional (default=16000)
        sample rate for samples.
    minimum_amplitude: int, optional (default=100)
        abs(minimum_amplitude) to assume a sample is a voice activity. Else, automatically False.

    Returns
    -------
    result : malaya_speech.model.webrtc.WebRTC class
    """

    try:
        import webrtcvad
    except BaseException:
        raise ModuleNotFoundError(
            'webrtcvad not installed. Please install it by `pip install webrtcvad` and try again.'
        )

    vad = webrtcvad.Vad(aggressiveness)
    return WebRTC(vad, sample_rate, minimum_amplitude)


@check_type
def deep_model(model: str = 'marblenet-factor1', quantized: bool = False, **kwargs):
    """
    Load VAD model.
    Prefer 50 ms or bigger frame.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Check available models at `malaya_speech.vad.available_model()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """

    warnings.warn(
        '`malaya_speech.vad.deep_model` is using Tensorflow, malaya-speech no longer improved it after version 1.4.0',
        DeprecationWarning)

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.vad.available_model()`.'
        )

    settings = {
        'vggvox-v1': {'frame_len': 0.005, 'frame_step': 0.0005},
        'vggvox-v2': {'hop_length': 24, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 1.0},
        'marblenet-factor1': {'feature_type': 'mfcc'},
        'marblenet-factor3': {'feature_type': 'mfcc'},
        'marblenet-factor5': {'feature_type': 'mfcc'},
    }

    return classification.load(
        model=model,
        module='vad',
        extra=settings[model],
        label=[False, True],
        quantized=quantized,
        **kwargs
    )


@check_type
def nemo(
    model: str = 'huseinzol05/nemo-vad-marblenet',
    **kwargs,
):
    """
    Load Nemo VAD model. 
    Nemo VAD prefer 63 ms frame, https://github.com/NVIDIA/NeMo/blob/02cf155b020964992a974e030b9e318426761e33/nemo/collections/asr/data/feature_to_label_dataset.py#L43

    Parameters
    ----------
    model : str, optional (default='huseinzol05/vad-marblenet')
        Check available models at `malaya_speech.vad.available_nemo()`.

    Returns
    -------
    result : malaya_speech.torch_model.nemo.Classification class
    """
    if model not in _nemo_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.vad.available_nemo()`.'
        )

    return classification.nemo_classification(
        model=model,
        label=[False, True],
        **kwargs
    )
