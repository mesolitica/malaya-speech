from typing import List, TextIO


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_srt(transcript: List[dict], file: TextIO):
    """
    Write list of transcription into SRT format.

    Parameters
    ----------
    transcript: List[dict]
        list of {'start', 'end', 'text'}
    file: typing.TextIO

    """

    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def write_vtt(transcript: List[dict], file: TextIO):
    """
    Write list of transcription into VTT format.

    Parameters
    ----------
    transcript: List[dict]
        list of {'start', 'end', 'text'}
    file: typing.TextIO

    """

    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def write_tsv(transcript: List[dict], file: TextIO):
    """
    Write list of transcription into TSV format.

    Parameters
    ----------
    transcript: List[dict]
        list of {'start', 'end', 'text'}
    file: typing.TextIO

    """

    print("start", "end", "text", sep="\t", file=file)
    for segment in transcript:
        print(segment['start'], file=file, end="\t")
        print(segment['end'], file=file, end="\t")
        print(segment['text'].strip().replace("\t", " "), file=file, flush=True)
