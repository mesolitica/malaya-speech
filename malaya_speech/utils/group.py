from malaya_speech.model.frame import FRAME
from typing import List


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
    result : List[FRAME]
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
