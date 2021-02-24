from malaya_speech.model.frame import Frame
from malaya_speech.utils.group import (
    combine_frames,
    group_frames,
    group_frames_threshold,
)


def split_vad(
    frames,
    n: int = 3,
    negative_threshold: float = 0.1,
    silent_trail: int = 500,
    sample_rate: int = 16000,
):
    """
    Split a sample into multiple samples based `n` size of negative VAD.

    Parameters
    ----------
    frames: List[Tuple[Frame, label]]
    n: int, optional (default=3)
        `n` size of negative VAD to assume in one subsample.
    negative_threshold: float, optional (default = 0.1)
        If `negative_threshold` is 0.1, means that, length negative samples must at least 0.1 second.
    silent_trail: int, optional (default = 500)
        If an element is not a voice activity, append with `silent_trail` frame size. 
    sample_rate: int, optional (default = 16000)
        sample rate for frames.

    Returns
    -------
    result : List[Frame]
    """
    grouped = group_frames(frames)
    grouped = group_frames_threshold(
        grouped, threshold_to_stop = negative_threshold
    )
    results, temp, not_activities = [], [], 0
    for no, g in enumerate(grouped):
        if g[1]:
            a = g[0]
        else:
            not_activities += 1
            a = np.concatenate(
                [g[0].array[:silent_trail], g[0].array[-silent_trail:]]
            )
            a = Frame(
                array = a,
                timestamp = g[0].timestamp,
                duration = len(a) / sample_rate,
            )
        temp.append(a)
        if not_activities >= n:
            results.append(combine_frames(temp))
            temp = [g[0]]
            not_activities = 0

    if len(temp):
        results.append(combine_frames(temp))
    return results


def split_vad_duration(
    frames,
    min_duration: float = 5.0,
    negative_threshold: float = 0.1,
    silent_trail = 500,
    sample_rate: int = 16000,
):
    """
    Split a sample into multiple samples based minimum duration of voice activities.

    Parameters
    ----------
    frames: List[Tuple[Frame, label]]
    min_duration: float, optional (default = 5.0)
        Minimum duration to assume one sample combined from voice activities.
    negative_threshold: float, optional (default = 0.1)
        If `negative_threshold` is 0.1, means that, length negative samples must at least 0.1 second.
    silent_trail: int, optional (default = 500)
        If an element is not a voice activity, append with `silent_trail` frame size.
    sample_rate: int, optional (default = 16000)
        sample rate for frames.

    Returns
    -------
    result : List[Frame]
    """
    grouped = group_frames(frames)
    grouped = group_frames_threshold(
        grouped, threshold_to_stop = negative_threshold
    )
    results, temp, lengths, last_silent = [], [], 0, None
    for no, g in enumerate(grouped):
        if g[1]:
            a = g[0]
        else:
            last_silent = g[0]
            a = np.concatenate(
                [g[0].array[:silent_trail], g[0].array[-silent_trail:]]
            )
            a = Frame(
                array = a,
                timestamp = g[0].timestamp,
                duration = len(a) / sample_rate,
            )
        l = len(a.array) / sample_rate
        lengths += l
        temp.append(a)
        if lengths >= min_duration:
            results.append(combine_frames(temp))
            temp = [last_silent] if last_silent else []
            lengths = 0

    if len(temp):
        results.append(combine_frames(temp))
    return results
