from .split import group_frames, group_frames_threshold
import numpy as np
from typing import List


def without_silent(
    frames, threshold_to_stop: float = 0.1, silent_trail: int = 500
):
    """
    Group multiple frames based on label and threshold to stop.

    Parameters
    ----------
    frames: List[Tuple[Frame, label]]
        Output from VAD.
    threshold_to_stop: float, optional (default = 0.1)
        If `threshold_to_stop` is 0.1, means that, length same label samples must at least 0.1 second.
    silent_trail: int, optional (default = 500)
        if detected a silent, will append first N frames and last N frames.

    Returns
    -------
    result : np.array
    """
    grouped = group_frames(frames)
    grouped = group_frames_threshold(grouped, threshold_to_stop)
    r = []
    for g in grouped:
        if g[1]:
            g = g[0].array
        else:
            g = np.concatenate(
                [g[0].array[:silent_trail], g[0].array[-silent_trail:]]
            )
        r.append(g)
    audio = np.concatenate(r)
    return audio
