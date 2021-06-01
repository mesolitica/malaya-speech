from malaya_speech.supervised import stt, lm
from herpetologist import check_type
import json

_transducer_availability = {
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.23582,
        'CER': 0.08771,
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.21718,
        'CER': 0.07562,
    },
    'large-conformer': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.21938,
        'CER': 0.07306,
    },
    'small-alconformer': {
        'Size (MB)': 18.8,
        'Quantized Size (MB)': 10.1,
        'WER': 0.30373,
        'CER': 0.12471,
    },
    'alconformer': {
        'Size (MB)': 38,
        'Quantized Size (MB)': 14.9,
        'WER': 0.25611,
        'CER': 0.09726,
    },
    'small-conformer-mixed': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.43149,
        'CER': 0.29467,
    },
    'conformer-mixed': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.35191,
        'CER': 0.23667,
    },
    'large-conformer-mixed': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.3359,
        'CER': 0.1989,
    },
}

_ctc_availability = {
    'wav2vec2-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.30463,
        'CER': 0.07633,
    },
    'wav2vec2-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.2765,
        'CER': 0.0705,
    },
}

_language_model_availability = {
    'bahasa': {
        'Size (MB)': 17,
        'Description': 'Gathered from malaya-speech ASR bahasa transcript',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-news': {
        'Size (MB)': 26,
        'Description': 'Gathered from malaya-speech bahasa ASR transcript + News (Random sample 300k sentences)',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-combined': {
        'Size (MB)': 64,
        'Description': 'Gathered from malaya-speech bahasa ASR transcript + Bahasa News + Bahasa Wikipedia',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 2 --prune 0 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
}


def available_ctc():
    """
    List available Encoder-CTC ASR models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_ctc_availability)


def available_language_model():
    """
    List available Language Model for CTC.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_language_model_availability)


def available_transducer():
    """
    List available Encoder-Transducer ASR models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_transducer_availability)


@check_type
def language_model(
    model: str = 'bahasa', alpha: float = 2.5, beta: float = 0.3, **kwargs
):
    """
    Load KenLM language model.

    Parameters
    ----------
    model : str, optional (default='bahasa')
        Model architecture supported. Allowed values:

        * ``'bahasa'`` - Gathered from malaya-speech ASR bahasa transcript.
        * ``'bahasa-news'`` - Gathered from malaya-speech ASR bahasa transcript + Bahasa News (Random sample 300k sentences).
        * ``'bahasa-combined'`` - Gathered from malaya-speech ASR bahasa transcript + Bahasa News + Bahasa Wikipedia.
        
    alpha: float, optional (default=2.5)
        score = alpha * np.log(lm) + beta * np.log(word_cnt), 
        increase will put more bias on lm score computed by kenlm.
    beta: float, optional (beta=0.3)
        score = alpha * np.log(lm) + beta * np.log(word_cnt), 
        increase will put more bias on word count.

    Returns
    -------
    result : Tuple[ctc_decoders.Scorer, List[str]]
        Tuple of ctc_decoders.Scorer and vocab.
    """
    model = model.lower()
    if model not in _language_model_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_language_model()`.'
        )

    scorer = lm.load(
        model = model,
        module = 'language-model',
        alpha = alpha,
        beta = beta,
        **kwargs
    )
    return scorer


@check_type
def deep_ctc(
    model: str = 'wav2vec2-conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='conformer')
        Model architecture supported. Allowed values:

        * ``'wav2vec2-conformer'`` - Finetuned Wav2Vec2 Conformer.
        * ``'wav2vec2-conformer-large'`` - Finetuned Wav2Vec2 Conformer LARGE.
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.CTC class
    """
    model = model.lower()
    if model not in _transducer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_ctc()`.'
        )

    return stt.wav2vec2_ctc_load(
        model = model,
        module = 'speech-to-text-ctc',
        quantized = quantized,
        **kwargs
    )


@check_type
def deep_transducer(
    model: str = 'conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='conformer')
        Model architecture supported. Allowed values:

        * ``'small-conformer'`` - SMALL size Google Conformer with Pretrained LM Malay language.
        * ``'conformer'`` - BASE size Google Conformer with Pretrained LM Malay language.
        * ``'large-conformer'`` - LARGE size Google Conformer with Pretrained LM Malay language.
        * ``'small-alconformer'`` - SMALL size A-Lite Google Conformer with Pretrained LM Malay language.
        * ``'alconformer'`` - BASE size A-Lite Google Conformer with Pretrained LM Malay language.
        * ``'small-conformer-mixed'`` - SMALL size Google Conformer with Pretrained LM Mixed (Malay + Singlish) languages.
        * ``'conformer-mixed'`` - BASE size Google Conformer with Pretrained LM Mixed (Malay + Singlish) languages.
        * ``'large-conformer-mixed'`` - LARGE size Google Conformer with Pretrained LM Mixed (Malay + Singlish) languages.
        
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.Transducer class
    """
    model = model.lower()
    if model not in _transducer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_transducer()`.'
        )

    if 'wav2vec2' in model:
        interface = stt.wav2vec_transducer_load
    else:
        interface = stt.transducer_load

    return interface(
        model = model,
        module = 'speech-to-text',
        quantized = quantized,
        **kwargs
    )
