from malaya_speech.supervised import stt
from malaya_speech.stt.transducer import available_pt_transformer, _pt_transformer_availability
from malaya_speech.utils import describe_availability
from herpetologist import check_type
import warnings

_transformer_availability = {
    'conformer-transducer': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.3,
        'Language': ['malay'],
    },
    'conformer-transducer-mixed': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.3,
        'Language': ['malay', 'singlish'],
    },
    'conformer-transducer-singlish': {
        'Size (MB)': 120,
        'Quantized Size (MB)': 32.3,
        'Language': ['singlish'],
    },
}


def available_transformer():
    """
    List available Encoder-Transducer Aligner models.
    """

    return describe_availability(_transformer_availability)


@check_type
def transformer(
    model: str = 'conformer-transducer',
    quantized: bool = False,
    **kwargs,
):
    """
    Load Encoder-Transducer Aligner model.

    Parameters
    ----------
    model : str, optional (default='conformer-transducer')
        Check available models at `malaya_speech.force_alignment.transducer.available_transformer()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.transducer.TransducerAligner class
    """

    warnings.warn(
        '`malaya.force_alignment.transducer.transformer` is using Tensorflow, means malaya-speech no longer improved it.',
        DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.transducer.available_transformer()`.'
        )

    return stt.transducer_load(
        model=model,
        module='force-alignment',
        languages=_transformer_availability[model]['Language'],
        quantized=quantized,
        stt=False,
        **kwargs
    )


def pt_transformer(
    model: str = 'mesolitica/conformer-base',
    **kwargs,
):
    """
    Load Encoder-Transducer ASR model using Pytorch.

    Parameters
    ----------
    model : str, optional (default='mesolitica/conformer-base')
        Check available models at `malaya_speech.force_alignment.transducer.available_pt_transformer()`.

    Returns
    -------
    result : malaya_speech.torch_model.torchaudio.ForceAlignment class
    """

    if model not in _pt_transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.transducer.available_pt_transformer()`.'
        )

    return stt.torchaudio(model=model, stt=False, **kwargs)
