from malaya_speech.supervised import unet
from malaya_speech.utils.astype import int_to_float
from herpetologist import check_type
import librosa
import numpy as np

# https://github.com/sigsep/sigsep-mus-eval/blob/master/museval/__init__.py#L364
# Only calculate SDR, ISR, SAR on voice sample

_availability = {
    'unet': {
        'Size (MB)': 78.9,
        'Quantized Size (MB)': 20,
        'SUM MAE': 0.862316,
        'MAE_SPEAKER': 0.460676,
        'MAE_NOISE': 0.401640,
        'SDR': 9.17312,
        'ISR': 13.92435,
        'SAR': 13.20592,
    },
    'resnet-unet': {
        'Size (MB)': 96.4,
        'Quantized Size (MB)': 24.6,
        'SUM MAE': 0.82535,
        'MAE_SPEAKER': 0.43885,
        'MAE_NOISE': 0.38649,
        'SDR': 9.45413,
        'ISR': 13.9639,
        'SAR': 13.60276,
    },
    'resnext-unet': {
        'Size (MB)': 75.4,
        'Quantized Size (MB)': 19,
        'SUM MAE': 0.81102,
        'MAE_SPEAKER': 0.44719,
        'MAE_NOISE': 0.363830,
        'SDR': 8.992832,
        'ISR': 13.49194,
        'SAR': 13.13210,
    },
}


def available_model():
    """
    List available Noise Reduction deep learning models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability,
        text = 'Only calculate SDR, ISR, SAR on voice sample. Higher is better.',
    )


@check_type
def deep_model(model: str = 'resnet-unet', quantized: bool = False, **kwargs):
    """
    Load Noise Reduction deep learning model.

    Parameters
    ----------
    model : str, optional (default='wavenet')
        Model architecture supported. Allowed values:

        * ``'unet'`` - pretrained UNET.
        * ``'resnet-unet'`` - pretrained resnet-UNET.
        * ``'resnext'`` - pretrained resnext-UNET.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.tf.UNET_STFT class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.noise_reduction.available_model()`.'
        )

    return unet.load_stft(
        model = model,
        module = 'noise-reduction',
        instruments = ['voice', 'noise'],
        quantized = quantized,
        **kwargs
    )


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_power(y, sr = 16000):
    from pysndfx import AudioEffectsChain

    y = int_to_float(y)
    cent = librosa.feature.spectral_centroid(y = y, sr = sr)

    threshold_h = round(np.median(cent)) * 1.5
    threshold_l = round(np.median(cent)) * 0.1

    less_noise = (
        AudioEffectsChain()
        .lowshelf(gain = -30.0, frequency = threshold_l, slope = 0.8)
        .highshelf(gain = -12.0, frequency = threshold_h, slope = 0.5)
    )
    y_clean = less_noise(y)

    return y_clean


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_centroid_s(y, sr = 16000):
    try:
        from pysndfx import AudioEffectsChain
    except Exception as e:
        raise ModuleNotFoundError(
            'pysndfx not installed. Please install it by `pip install pysndfx` and try again.'
        )

    y = int_to_float(y)
    cent = librosa.feature.spectral_centroid(y = y, sr = sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = (
        AudioEffectsChain()
        .lowshelf(gain = -12.0, frequency = threshold_l, slope = 0.5)
        .highshelf(gain = -12.0, frequency = threshold_h, slope = 0.5)
        .limiter(gain = 6.0)
    )

    y_cleaned = less_noise(y)

    return y_cleaned


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_centroid_mb(y, sr = 16000):
    try:
        from pysndfx import AudioEffectsChain
    except Exception as e:
        raise ModuleNotFoundError(
            'pysndfx not installed. Please install it by `pip install pysndfx` and try again.'
        )

    y = int_to_float(y)
    cent = librosa.feature.spectral_centroid(y = y, sr = sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = (
        AudioEffectsChain()
        .lowshelf(gain = -30.0, frequency = threshold_l, slope = 0.5)
        .highshelf(gain = -30.0, frequency = threshold_h, slope = 0.5)
        .limiter(gain = 10.0)
    )
    y_cleaned = less_noise(y)

    cent_cleaned = librosa.feature.spectral_centroid(y = y_cleaned, sr = sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows / 3 * 2)
    boost_l = math.floor(rows / 6)
    boost = math.floor(rows / 3)
    boost_bass = AudioEffectsChain().lowshelf(
        gain = 16.0, frequency = boost_h, slope = 0.5
    )
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_mfcc_down(y, sr = 16000):
    try:
        from pysndfx import AudioEffectsChain
        import python_speech_features
    except Exception as e:
        raise ModuleNotFoundError(
            'pysndfx, python_speech_features not installed. Please install it by `pip install pysndfx python_speech_features` and try again.'
        )

    y = int_to_float(y)
    hop_length = 512

    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n ** 2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = (
        AudioEffectsChain()
        .highshelf(frequency = min_hz * (-1) * 1.2, gain = -12.0, slope = 0.6)
        .limiter(gain = 8.0)
    )
    y_speach_boosted = speech_booster(y)

    return y_speach_boosted


# https://github.com/dodiku/noise_reduction/blob/master/noise.py
def reduce_noise_mfcc_up(y, sr = 16000):
    try:
        from pysndfx import AudioEffectsChain
        import python_speech_features
    except Exception as e:
        raise ModuleNotFoundError(
            'pysndfx, python_speech_features not installed. Please install it by `pip install pysndfx python_speech_features` and try again.'
        )

    y = int_to_float(y)
    hop_length = 512
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n ** 2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(
        frequency = min_hz * (-1), gain = 12.0, slope = 0.5
    )
    y_speach_boosted = speech_booster(y)

    return y_speach_boosted


def trim_silence(
    y,
    top_db = 20,
    frame_length = 2,
    hop_length = 500,
    return_trimmed_length = False,
):
    y = int_to_float(y)
    y_trimmed, index = librosa.effects.trim(
        y, top_db = top_db, frame_length = frame_length, hop_length = hop_length
    )
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)

    if return_trimmed_length:
        return y_trimmed, trimmed_length
    else:
        return y_trimmed
