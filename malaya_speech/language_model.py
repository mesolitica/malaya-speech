from malaya_speech.utils import describe_availability
from malaya_speech.supervised import lm
from malaya_speech.torch_model.mask_lm import LM as Mask_LM
from herpetologist import check_type

_kenlm_availability = {
    'bahasa-news': {
        'Size (MB)': 107,
        'LM order': 3,
        'Description': 'local news.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-wiki': {
        'Size (MB)': 70.5,
        'LM order': 3,
        'Description': 'MS wikipedia.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'redape-community': {
        'Size (MB)': 887.1,
        'LM order': 4,
        'Description': 'Mirror for https://github.com/redapesolutions/suara-kami-community',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 4 --prune 0 1 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'dump-combined': {
        'Size (MB)': 310,
        'LM order': 3,
        'Description': 'Academia + News + IIUM + Parliament + Watpadd + Wikipedia + Common Crawl + training set from https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'manglish': {
        'Size (MB)': 202,
        'LM order': 3,
        'Description': 'Manglish News + Manglish Reddit + Manglish forum + training set from https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-manglish-combined': {
        'Size (MB)': 608,
        'LM order': 3,
        'Description': 'Combined `dump-combined` and `manglish`.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
}

_gpt2_availability = {
    'mesolitica/gpt2-117m-bahasa-cased': {
        'Size (MB)': 454,
    },
}

_mlm_availability = {
    'mesolitica/bert-base-standard-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/roberta-base-standard-bahasa-cased': {
        'Size (MB)': 422,
    },
}


def available_kenlm():
    """
    List available KenLM Language Model.
    """

    return describe_availability(_kenlm_availability)


def available_gpt2():
    """
    List available GPT2 Language Model.
    """

    return describe_availability(_gpt2_availability)


def available_mlm():
    """
    List available MLM Language Model.
    """

    return describe_availability(_mlm_availability)


@check_type
def kenlm(
    model: str = 'dump-combined', **kwargs
):
    """
    Load KenLM language model.

    Parameters
    ----------
    model : str, optional (default='dump-combined')
        Check available models at `malaya_speech.language_model.available_kenlm()`.

    Returns
    -------
    result : str
    """
    model = model.lower()
    if model not in _kenlm_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.language_model.available_kenlm()`.'
        )

    path_model = lm.load(
        model=model,
        module='language-model',
        **kwargs
    )
    return path_model


@check_type
def gpt2(model: str = 'mesolitica/gpt2-117m-bahasa-cased', force_check: bool = True, **kwargs):
    """
    Load GPT2 language model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/gpt2-117m-bahasa-cased')
        Check available models at `malaya_speech.language_model.available_gpt2()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.gpt2_lm.LM class
    """

    if model not in _gpt2_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.language_model.available_gpt2()`.'
        )

    return lm.gpt2_load(model, **kwargs)


@check_type
def mlm(model: str = 'mesolitica/bert-base-standard-bahasa-cased', force_check: bool = True, **kwargs):
    """
    Load Masked language model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/bert-base-standard-bahasa-cased')
        Check available models at `malaya_speech.language_model.available_mlm()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya_speech.torch_model.mask_lm.LM class
    """

    if model not in _mlm_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya_speech.language.available_mlm()`.'
        )

    try:
        import malaya
    except BaseException:
        raise ModuleNotFoundError(
            'malaya not installed. Please install it by `pip install malaya` and try again.'
        )

    try:
        mask_lm = malaya.language_model.mlm(model=model, force_check=False)
    except BaseException:
        raise ModuleNotFoundError(
            'required malaya >= 4.9.2. Please update malaya version by `pip install malaya -U` and try again.'
        )

    return Mask_LM(mask_lm, **kwargs)
