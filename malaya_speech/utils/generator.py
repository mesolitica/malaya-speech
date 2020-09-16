from malaya_speech.model.frame import FRAME
from herpetologist import check_type


@check_type
def frames(
    audio,
    frame_duration_ms: int = 30,
    sample_rate: int = 16000,
    append_ending_trail: bool = True,
):
    """
    Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
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
