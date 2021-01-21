from malaya_speech.model.frame import Frame
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
    Generates audio frames from audio.
    Takes the desired frame duration in milliseconds, the audio, and the sample rate.

    Parameters
    ----------
    audio: np.array
    frame_duration_ms: int, optional (default=30)
    sample_rate: int, optional (default=16000)
    append_ending_trail: bool, optional (default=True)
        if True, will append last trail and this last trail might not same length as `frame_duration_ms`.

    Returns
    -------
    result: List[malaya_speech.model.frame.Frame]
    """

    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    timestamp = 0.0
    duration = float(n) / sample_rate
    results = []
    while offset + n <= len(audio):
        results.append(Frame(audio[offset : offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    if append_ending_trail and offset < len(audio):
        results.append(
            Frame(
                audio[offset:], timestamp, len(audio) / sample_rate - timestamp
            )
        )
    return results


def mel_sampling(
    audio, frame_duration_ms = 1200, overlap_ms = 200, sample_rate = 16000
):
    """
    Generates audio frames from audio. This is for melspectrogram generative model.
    Takes the desired frame duration in milliseconds, the audio, and the sample rate.

    Parameters
    ----------
    audio: np.array
    frame_duration_ms: int, optional (default=1200)
    overlap_ms: int, optional (default=200)
    sample_rate: int, optional (default=16000)

    Returns
    -------
    result: List[np.array]
    """

    n = int(sample_rate * (frame_duration_ms / 1000.0))
    n_overlap = int(sample_rate * (overlap_ms / 1000.0))
    offset = 0
    results = []
    while offset + n <= len(audio):
        results.append(audio[offset : offset + n])
        offset += n - n_overlap
    if offset < len(audio):
        results.append(audio[offset:])

    return results


def combine_mel_sampling(
    samples, overlap_ms = 200, sample_rate = 16000, padded_ms = 50
):
    """
    To combine results from `mel_sampling`, output from melspectrogram generative model.

    Parameters
    ----------
    samples: List[np.array]
    overlap_ms: int, optional (default=200)
    sample_rate: int, optional (default=16000)

    Returns
    -------
    result: List[np.array]
    """
    n_overlap = int(sample_rate * (overlap_ms / 1000.0))
    n_padded = int(sample_rate * (padded_ms / 1000.0))
    results = []
    for no, sample in enumerate(samples):
        if no:
            sample = sample[n_overlap - n_padded :]
        results.append(sample[:-n_padded])
    return results
