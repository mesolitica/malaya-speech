from .group import combine_frames, group_frames, group_frames_threshold


def split_vad(frames, n = 3, negative_threshold: float = 0.1):
    """
    Split a sample into multiple samples based `n` size of negative VAD.

    Parameters
    ----------
    frames: List[Tuple[Frame, label]]
    n: int, optional (default=3)
        `n` size of negative VAD to assume in one subsample.
    negative_threshold: float, optional (default = 0.1)
        If `negative_threshold` is 0.1, means that, length negative samples must at least 0.1 second.

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
        if not g[1]:
            not_activities += 1
        temp.append(g[0])
        if not_activities >= n:
            results.append(combine_frames(temp))
            temp = [g[0]]
            not_activities = 0

    if len(temp):
        results.append(combine_frames(temp))
    return results
