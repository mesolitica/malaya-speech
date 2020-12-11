from malaya.utils.text import (
    convert_to_ascii,
    collapse_whitespace,
    put_spacing_num,
)

_tacotron2_availability = {'male': {}, 'female': {}}
_fastspeech2_availability = {'male': {}, 'female': {}}

_pad = 'pad'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


MALAYA_SPEECH_SYMBOLS = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + [_eos]
)


def tts_encode(string: str, add_eos: bool = True):
    r = [MALAYA_SPEECH_SYMBOLS.index(c) for c in string]
    if add_eos:
        r = r + [MALAYA_SPEECH_SYMBOLS.index('eos')]
    return r


class NORMALIZER:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def normalize(
        self, string, normalize = True, lowercase = True, add_eos = True
    ):
        string = convert_to_ascii(string)
        string = string.replace('&', ' dan ')
        string = put_spacing_num(string)
        if normalize:
            string = normalizer.normalize(
                string,
                check_english = False,
                normalize_entity = False,
                normalize_text = False,
                normalize_url = True,
                normalize_email = True,
            )
            string = string['normalize']
        else:
            string = string
        if lowercase:
            string = string.lower()
        return string, tts_encode(string, add_eos = add_eos)


def load_normalizer():
    try:
        import malaya
    except:
        raise ModuleNotFoundError(
            'malaya not installed. Please install it by `pip install malaya` and try again.'
        )

    normalizer = malaya.normalize.normalizer(date = False, time = False)
    return NORMALIZER(normalizer)


def available_tacotron2():
    """
    List available Tacotron2, Text to Mel models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_tacotron2_availability)


def available_fastspeech2():
    """
    List available FastSpeech2, Text to Mel models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(_fastspeech2_availability)


def tacotron2(model: str = 'male', quantized: bool = False, **kwargs):
    model = model.lower()

    if model not in _tacotron2_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.tts.available_tacotron2()`.'
        )

    normalizer = load_normalizer()


def fastspeech2(model: str = 'male', quantized: bool = False, **kwargs):
    model = model.lower()

    if model not in _fastspeech2_availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.tts.available_fastspeech2()`.'
        )

    normalizer = load_normalizer()
