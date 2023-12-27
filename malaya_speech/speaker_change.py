from malaya_speech.supervised import classification
from malaya_speech.model.frame import Frame
import logging

logger = logging.getLogger(__name__)

_nemo_availability = {
    'huseinzol05/nemo-is-clean-speakernet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet',
        'Size (MB)': 16.2,
    },
    'huseinzol05/nemo-is-clean-titanet_large': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet',
        'Size (MB)': 88.8,
    },
}


def split_activities(
    vad_results,
    speaker_change_results,
    speaker_change_threshold: float = 0.5,
    sr: int = 16000,
    ignore_not_activity=True,
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
                            int(before * sr): int((before + after) * sr)
                        ],
                        before_timestamp,
                        after,
                    )
                    results.append((f, result[1]))
                    before = after
                    before_timestamp = t

                if result[0].timestamp + result[0].duration > before_timestamp:
                    f = Frame(
                        result[0].array[int(before * sr):],
                        before_timestamp,
                        (result[0].timestamp + result[0].duration)
                        - before_timestamp,
                    )
                    results.append((f, result[1]))

            else:
                results.append(result)
    return results
