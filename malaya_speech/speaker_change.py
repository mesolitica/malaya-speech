from malaya_speech.path import PATH_SPEAKER_CHANGE, S3_PATH_SPEAKER_CHANGE
from malaya_speech.supervised import classification
from malaya_speech.model.frame import Frame
from herpetologist import check_type

_availability = {
    'vggvox-v2': {
        'Size (MB)': 31.1,
        'Quantized Size (MB)': 7.92,
        'Accuracy': 0.63979,
    },
    'speakernet': {
        'Size (MB)': 20.3,
        'Quantized Size (MB)': 5.18,
        'Accuracy': 0.64524,
    },
}


def available_model():
    """
    List available speaker change deep models.
    """
    from malaya_speech.utils import describe_availability

    return describe_availability(
        _availability,
        text = 'last accuracy during training session before early stopping.',
    )


@check_type
def deep_model(model: str = 'speakernet', quantized: bool = False, **kwargs):
    """
    Load speaker change deep model.

    Parameters
    ----------
    model : str, optional (default='vggvox-v2')
        Model architecture supported. Allowed values:

        * ``'vggvox-v2'`` - finetuned VGGVox V2.
        * ``'speakernet'`` - finetuned SpeakerNet.
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.supervised.classification.load function
    """
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from `malaya_speech.speaker_change.available_model()`.'
        )

    settings = {
        'vggvox-v2': {'hop_length': 50, 'concat': False, 'mode': 'eval'},
        'speakernet': {'frame_ms': 20, 'stride_ms': 2},
    }

    return classification.load(
        path = PATH_SPEAKER_CHANGE,
        s3_path = S3_PATH_SPEAKER_CHANGE,
        model = model,
        name = 'speaker-change',
        extra = settings[model],
        label = [False, True],
        quantized = quantized,
        **kwargs
    )


def split_activities(
    vad_results,
    speaker_change_results,
    speaker_change_threshold: float = 0.5,
    sr: int = 16000,
    ignore_not_activity = True,
):
    """
    split VAD based on speaker change threshold, worse-case O(N^2).

    Parameters
    ----------
    vad_results: List[Tuple[Frame, label]]
        results from VAD.
    speaker_change_results: List[Tuple[Frame, float]], optional (default=None)
        results from speaker change module, must in float result.
    speaker_change_threshold: float, optional (default=0.5)
        in one voice activity sample can be more than one speaker, split it using this threshold.
    sr: int, optional (default=16000)
        sample rate, classification model in malaya-speech use 16k.
    ignore_not_activity: bool, optional (default=True)
        If True, will ignore if result VAD is False, else will try to split.

    Returns
    -------
    result : List[Tuple[Frame, label]]
    """

    if not 0 < speaker_change_threshold <= 1.0:
        raise ValueError(
            'speaker_change_threshold must, 0 < speaker_change_threshold <= 1.0'
        )

    results = []
    for result in vad_results:
        if not result[1] and ignore_not_activity:
            results.append(result)
        else:
            group = []
            for change in speaker_change_results:
                from_vad = result[0].timestamp
                until_vad = result[0].duration + from_vad

                from_change = change[0].timestamp
                until_change = (change[0].duration / 2) + from_change

                change_result = change[1]
                if (
                    until_change >= from_vad
                    and until_change <= until_vad
                    and change_result >= speaker_change_threshold
                ):
                    group.append(until_change)
            if len(group):
                before = 0
                before_timestamp = result[0].timestamp
                for t in group:
                    after = t - before_timestamp
                    f = Frame(
                        result[0].array[
                            int(before * sr) : int((before + after) * sr)
                        ],
                        before_timestamp,
                        after,
                    )
                    results.append((f, result[1]))
                    before = after
                    before_timestamp = t

                if result[0].timestamp + result[0].duration > before_timestamp:
                    f = Frame(
                        result[0].array[int(before * sr) :],
                        before_timestamp,
                        (result[0].timestamp + result[0].duration)
                        - before_timestamp,
                    )
                    results.append((f, result[1]))

            else:
                results.append(result)
    return results
