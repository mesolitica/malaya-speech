from malaya_speech.supervised import stt, lm
from herpetologist import check_type
import json

_transducer_availability = {
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.20599,
        'CER': 0.08933,
        'Language': ['malay'],
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.16547,
        'CER': 0.06410,
        'Language': ['malay'],
    },
    'large-conformer': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.15986,
        'CER': 0.05937,
        'Language': ['malay'],
    },
    'alconformer': {
        'Size (MB)': 38.1,
        'Quantized Size (MB)': 15.1,
        'WER': 0.20703,
        'CER': 0.08533,
        'Language': ['malay'],
    },
    'conformer-mixed': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.25314,
        'CER': 0.15836,
        'Language': ['malay', 'singlish'],
    },
    'large-conformer-mixed': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.24829,
        'CER': 0.16606,
        'Language': ['malay', 'singlish'],
    },
}

_ctc_availability = {
    'wav2vec2-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.25899,
        'CER': 0.06350,
        'Language': ['malay'],
    },
    'wav2vec2-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.23997,
        'CER': 0.05827,
        'Language': ['malay'],
    },
}

google_accuracy = {
    'malay': {
        'WER': 0.164775,
        'CER': 0.0597320
    }
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
        'Size (MB)': 24,
        'Description': 'Gathered from malaya-speech bahasa ASR transcript + News (Random sample 300k sentences)',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-combined': {
        'Size (MB)': 29,
        'Description': 'Gathered from malaya-speech ASR bahasa transcript + Bahasa News (Random sample 300k sentences) + Bahasa Wikipedia (Random sample 150k sentences).',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 2 --prune 0 1 1',
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
        * ``'bahasa-combined'`` - Gathered from malaya-speech ASR bahasa transcript + Bahasa News (Random sample 300k sentences) + Bahasa Wikipedia (Random sample 150k sentences).

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
        model=model,
        module='language-model',
        alpha=alpha,
        beta=beta,
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
    if model not in _ctc_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_ctc()`.'
        )

    return stt.wav2vec2_ctc_load(
        model=model,
        module='speech-to-text-ctc',
        quantized=quantized,
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

        * ``'small-conformer'`` - SMALL size Google Conformer.
        * ``'conformer'`` - BASE size Google Conformer.
        * ``'large-conformer'`` - LARGE size Google Conformer.
        * ``'small-alconformer'`` - SMALL size A-Lite Google Conformer.
        * ``'alconformer'`` - BASE size A-Lite Google Conformer.
        * ``'small-conformer-mixed'`` - SMALL size Google Conformer for (Malay + Singlish) languages.
        * ``'conformer-mixed'`` - BASE size Google Conformer for (Malay + Singlish) languages.
        * ``'large-conformer-mixed'`` - LARGE size Google Conformer for (Malay + Singlish) languages.

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

    return stt.transducer_load(
        model=model,
        module='speech-to-text-transducer',
        quantized=quantized,
        **kwargs
    )
