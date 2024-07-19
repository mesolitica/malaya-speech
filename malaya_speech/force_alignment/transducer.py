from malaya_speech.supervised import stt
from malaya_speech.stt.transducer import available_huggingface


def huggingface(
    model: str = 'mesolitica/conformer-base',
    force_check: bool = True,
    **kwargs,
):
    """
    Load Encoder-Transducer ASR model using Pytorch.

    Parameters
    ----------
    model : str, optional (default='mesolitica/conformer-base')
        Check available models at `malaya_speech.force_alignment.transducer.available_pt_transformer()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result : malaya_speech.torch_model.torchaudio.ForceAlignment class
    """

    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.force_alignment.transducer.available_huggingface`.'
        )

    return stt.torchaudio(model=model, stt=False, **kwargs)
