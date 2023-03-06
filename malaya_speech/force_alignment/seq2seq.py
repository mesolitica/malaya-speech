from malaya_speech.supervised import stt
from malaya_speech.stt.seq2seq import available_huggingface, _huggingface_availability
from malaya_speech.utils import describe_availability


def huggingface(
    model: str = 'mesolitica/finetune-whisper-base-ms-singlish-v2',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetune-whisper-base-ms-singlish-v2')
        Check available models at `malaya_speech.force_alignment.seq2seq.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.model.huggingface.Seq2SeqAligner class
    """

    model = model.lower()
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.seq2seq.available_huggingface()`.'
        )

    return stt.huggingface_load_seq2seq(model=model, stt=False, **kwargs)
