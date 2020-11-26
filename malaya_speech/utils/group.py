from malaya_speech.model.frame import FRAME
from collections import defaultdict
from typing import List
import operator
import numpy as np


def combine_frames(frames: List[FRAME]):
    """
    Combine multiple frames into one frame.

    Parameters
    ----------
    frames: List[FRAME]

    Returns
    -------
    result : FRAME
    """
    a, duration = [], 0
    for r in frames:
        a.extend(r.array)
        duration += r.duration
    return FRAME(a, frames[0].timestamp, duration)


def group_frames(frames):
    """
    Group multiple frames based on label.

    Parameters
    ----------
    frames: List[Tuple[FRAME, label]]

    Returns
    -------
    result : List[Tuple[FRAME, label]]
    """
    results, result, last = [], [], None

    for frame in frames:
        if last is None:
            last = frame[1]
            result.append(frame[0])
        elif last == frame[1]:
            result.append(frame[0])
        else:
            a, duration = [], 0
            for r in result:
                a.extend(r.array)
                duration += r.duration
            results.append((FRAME(a, result[0].timestamp, duration), last))
            result = [frame[0]]
            last = frame[1]

    if len(result):
        a, duration = [], 0
        for r in result:
            a.extend(r.array)
            duration += r.duration
        results.append((FRAME(a, result[0].timestamp, duration), last))
    return results


def group_frames_threshold(frames, threshold_to_stop: float = 0.3):
    """
    Group multiple frames based on label and threshold to stop.

    Parameters
    ----------
    frames: List[Tuple[FRAME, label]]
    threshold_to_stop: float, optional (default = 0.3)
        If `threshold_to_stop` is 0.3, means that, length same label samples must at least 0.3 second.

    Returns
    -------
    result : List[Tuple[FRAME, label]]
    """
    d = defaultdict(float)

    label, results, result = None, [], []
    for i in frames:
        d[i[1]] += i[0].duration
        result.append(i[0])
        if i[0].duration > threshold_to_stop:
            a = np.concatenate([i.array for i in result])
            durations = sum([i.duration for i in result])
            results.append(
                (
                    FRAME(a, result[0].timestamp, durations),
                    max(d.items(), key = operator.itemgetter(1))[0],
                )
            )
            d = defaultdict(float)
            result = []

    if len(result):
        a = np.concatenate([i.array for i in result])
        durations = sum([i.duration for i in result])
        results.append(
            (
                FRAME(a, result[0].timestamp, durations),
                max(d.items(), key = operator.itemgetter(1))[0],
            )
        )

    return results


def min_max_boundary(i, scale):
    minimum = i * scale
    maximum = (i + 1) * scale
    return int(minimum), int(maximum)


# minimum, maximum = min_max_boundary(0, 391520 / 241)
# int(0 * 241 / 391520)
