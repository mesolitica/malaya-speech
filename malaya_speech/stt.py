from malaya_speech.supervised import stt, lm
from herpetologist import check_type
import json

_transducer_availability = {
    'tiny-conformer': {
        'Size (MB)': 24.4,
        'Quantized Size (MB)': 9.14,
        'WER': 0.2128108,
        'CER': 0.08136871,
        'Language': ['malay'],
    },
    'small-conformer': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.1981407,
        'CER': 0.075313,
        'Language': ['malay'],
    },
    'conformer': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.1636023,
        'CER': 0.0587443,
        'Language': ['malay'],
    },
    'large-conformer': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.15986,
        'CER': 0.05937,
        'Language': ['malay'],
    },
    'conformer-mixed': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.279536,
        'CER': 0.177017,
        'Language': ['malay', 'singlish'],
    },
    'large-conformer-mixed': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.262179,
        'CER': 0.165555,
        'Language': ['malay', 'singlish'],
    },
    'conformer-stack-mixed': {
        'Size (MB)': 130,
        'Quantized Size (MB)': 38.5,
        'WER': 0.2401982,
        'CER': 0.1577375,
        'Language': ['malay', 'singlish'],
    },
    'conformer-stack-3mixed': {
        'Size (MB)': 130,
        'Quantized Size (MB)': 38.5,
        'WER': 0.2401982,
        'CER': 0.1577375,
        'Language': ['malay', 'singlish', 'mandarin'],
    },
    'small-conformer-singlish': {
        'Size (MB)': 49.2,
        'Quantized Size (MB)': 18.1,
        'WER': 0.12771,
        'CER': 0.0703953,
        'Language': ['singlish'],
    },
    'conformer-singlish': {
        'Size (MB)': 125,
        'Quantized Size (MB)': 37.1,
        'WER': 0.0963391,
        'CER': 0.0545533,
        'Language': ['singlish'],
    },
    'large-conformer-singlish': {
        'Size (MB)': 404,
        'Quantized Size (MB)': 107,
        'WER': 0.0839525,
        'CER': 0.0445617,
        'Language': ['singlish'],
    },
}

_ctc_availability = {
    'wav2vec2-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.255762,
        'CER': 0.061953,
        'WER-LM': 0.255762,
        'CER-LM': 0.061953,
        'Language': ['malay'],
    },
    'wav2vec2-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.23997,
        'CER': 0.05827,
        'WER-LM': 0.255762,
        'CER-LM': 0.061953,
        'Language': ['malay'],
    },
    'hubert-conformer-tiny': {
        'Size (MB)': 36.6,
        'Quantized Size (MB)': 10.3,
        'WER': 0.381819,
        'CER': 0.100910,
        'WER-LM': 0.20345644,
        'CER-LM': 0.06419384,
        'Language': ['malay'],
    },
    'hubert-conformer': {
        'Size (MB)': 115,
        'Quantized Size (MB)': 31.1,
        'WER': 0.255762,
        'CER': 0.061953,
        'WER-LM': 0.255762,
        'CER-LM': 0.061953,
        'Language': ['malay'],
    },
    'hubert-conformer-large': {
        'Size (MB)': 392,
        'Quantized Size (MB)': 100,
        'WER': 0.22837,
        'CER': 0.054309,
        'WER-LM': 0.255762,
        'CER-LM': 0.061953,
        'Language': ['malay'],
    },
}

google_accuracy = {
    'malay': {
        'WER': 0.164775,
        'CER': 0.0597320
    },
    'singlish': {
        'WER': 0.4941349,
        'CER': 0.3026296
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
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'redape-community': {
        'Size (MB)': 887.1,
        'Description': 'Mirror for https://github.com/redapesolutions/suara-kami-community',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 4 --prune 0 1 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'dump-combined': {
        'Size (MB)': 310,
        'Description': 'Academia + News + IIUM + Parliament + Watpadd + Wikipedia + Common Crawl + training set from https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
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
    model: str = 'dump-combined', **kwargs
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
        * ``'redape-community'`` - Mirror for https://github.com/redapesolutions/suara-kami-community
        * ``'dump-combined'`` - Academia + News + IIUM + Parliament + Watpadd + Wikipedia + Common Crawl + training set from https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean.

    Returns
    -------
    result : str
    """
    model = model.lower()
    if model not in _language_model_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.stt.available_language_model()`.'
        )

    path_model = lm.load(
        model=model,
        module='language-model',
        **kwargs
    )
    return path_model


@check_type
def deep_ctc(
    model: str = 'hubert-conformer', quantized: bool = False, **kwargs
):
    """
    Load Encoder-Transducer ASR model.

    Parameters
    ----------
    model : str, optional (default='hubert-conformer')
        Model architecture supported. Allowed values:

        * ``'wav2vec2-conformer'`` - Finetuned Wav2Vec2 Conformer.
        * ``'wav2vec2-conformer-large'`` - Finetuned Wav2Vec2 Conformer LARGE.
        * ``'hubert-conformer-tiny'`` - Finetuned HuBERT Conformer TINY.
        * ``'hubert-conformer'`` - Finetuned HuBERT Conformer.
        * ``'hubert-conformer-large'`` - Finetuned HuBERT Conformer LARGE.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.Wav2Vec2_CTC class
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
        mode=_ctc_availability[model]['mode'],
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

        * ``'tiny-conformer'`` - TINY size Google Conformer.
        * ``'small-conformer'`` - SMALL size Google Conformer.
        * ``'conformer'`` - BASE size Google Conformer.
        * ``'large-conformer'`` - LARGE size Google Conformer.
        * ``'conformer-mixed'`` - BASE size Google Conformer for (Malay + Singlish) languages.
        * ``'large-conformer-mixed'`` - LARGE size Google Conformer for (Malay + Singlish) languages.
        * ``'conformer-stack-mixed'`` - BASE size Stacked Google Conformer for (Malay + Singlish) languages.
        * ``'small-conformer-singlish'`` - SMALL size Google Conformer for singlish language.
        * ``'conformer-singlish'`` - BASE size Google Conformer for singlish language.
        * ``'large-conformer-singlish'`` - LARGE size Google Conformer for singlish language.

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
        languages=_transducer_availability[model]['Language'],
        quantized=quantized,
        **kwargs
    )
