from malaya_speech.model.interface import FRAME


def frames(frame_duration_ms, audio, sample_rate):
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
    while offset + n < len(audio):
        yield FRAME(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n
