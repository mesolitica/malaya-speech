from malaya_speech.supervised import stt
from malaya_speech.stt.seq2seq import _huggingface_availability
from malaya_speech.utils import describe_availability
from herpetologist import check_type
import warnings


def available_huggingface():
    """
    List available HuggingFace Malaya-Speech Aligner models.
    """

    return describe_availability(_huggingface_availability)


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-whisper-base-ms-singlish',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Finetuned models from HuggingFace.

    Parameters
    ----------
    model : str, optional (default='mesolitica/finetune-whisper-base-ms-singlish')
        Check available models at `malaya_speech.force_alignment.seq2seq.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.model.huggingface.Seq2SeqAligner class
    """

    model = model.lower()
    if model not in _huggingface_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.seq2seq.available_huggingface()`.'
        )

    return stt.huggingface_load_seq2seq(model=model, stt=False, **kwrags)
