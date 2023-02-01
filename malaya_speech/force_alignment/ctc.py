from malaya_speech.supervised import stt
from malaya_speech.stt.ctc import _transformer_availability, _huggingface_availability
from malaya_speech.utils import describe_availability
from herpetologist import check_type
import warnings


def available_transformer():
    """
    List available Encoder-CTC Aligner models.
    """
    warnings.warn(
        '`malaya.force_alignment.ctc.available_transformer` is deprecated, use `malaya.force_alignment.available_huggingface` instead', DeprecationWarning)

    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available HuggingFace Malaya-Speech Aligner models.
    """

    return describe_availability(_huggingface_availability)


@check_type
def transformer(
    model: str = 'hubert-conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-CTC ASR model.

    Parameters
    ----------
    model : str, optional (default='hubert-conformer')
        Check available models at `malaya_speech.force_alignment.ctc.available_transformer()`.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.wav2vec.Wav2Vec2_Aligner class
    """
    warnings.warn(
        '`malaya.force_alignment.ctc.transformer` is deprecated, use `malaya.force_alignment.ctc.huggingface` instead', DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.ctc.available_transformer()`.'
        )

    return stt.wav2vec2_ctc_load(
        model=model,
        module='speech-to-text-ctc-v2',
        quantized=quantized,
        mode=_transformer_availability[model],
        stt=False,
        **kwargs
    )


@check_type
def huggingface(
    model: str = 'mesolitica/wav2vec2-xls-r-300m-mixed',
    force_check: bool = True,
    **kwargs
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/wav2vec2-xls-r-300m-mixed')
        Check available models at `malaya_speech.force_alignment.ctc.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.model.huggingface.Aligner class
    """

    model = model.lower()
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.ctc.available_huggingface()`.'
        )

    return stt.huggingface_load(model=model, stt=False, **kwargs)
