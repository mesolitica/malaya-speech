from malaya_speech.supervised import stt
from malaya_speech.stt.ctc import available_huggingface


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
        Check available models at `malaya_speech.force_alignment.ctc.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.model.huggingface.Aligner class
    """

    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.ctc.available_huggingface()`.'
        )

    return stt.huggingface_load(model=model, stt=False, **kwargs)
