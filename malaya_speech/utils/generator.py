from malaya_speech.model.frame import FRAME
from herpetologist import check_type
from typing import List


@check_type
def frames(
    audio,
    frame_duration_ms: int = 30,
    sample_rate: int = 16000,
    append_ending_trail: bool = True,
):
    """
    Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and the sample rate.

    Parameters
    ----------

    audio: np.array / list
    frame_duration_ms: int, optional (default=30)
    sample_rate: int, optional (default=16000)
    append_ending_trail: bool, optional (default=True)
        if True, will append last trail and this last trail might not same length as `frame_duration_ms`.

    Returns
    -------
    result: List[malaya_speech.model.frame.FRAME]
    """

    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    timestamp = 0.0
    duration = float(n) / sample_rate
    results = []
    while offset + n < len(audio):
        results.append(FRAME(audio[offset : offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    if append_ending_trail:
        results.append(
            FRAME(
                audio[offset:], timestamp, len(audio) / sample_rate - timestamp
            )
        )
    return results


def combine_frames(frames: List[FRAME]):
    a, duration = [], 0
    for r in frames:
        a.extend(r.array)
        duration += r.duration
    return FRAME(a, frames[0].timestamp, duration)
