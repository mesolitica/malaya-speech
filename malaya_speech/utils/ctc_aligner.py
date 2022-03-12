import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from malaya_speech.utils.char import CTC_VOCAB as labels

# heavily copy from https://pytorch.org/audio/0.11.0/tutorials/forced_alignment_tutorial.html#generate-frame-wise-label-probability


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def get_trellis(emission, tokens, blank_id=len(labels)):
    num_frame = emission.shape[0]
    num_tokens = len(tokens)

    trellis = np.full((num_frame + 1, num_tokens + 1), -float('inf'))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = np.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=len(labels)):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.shape[1] - 1
    t_start = np.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = np.exp(emission[t - 1, tokens[j - 1] if changed > stayed else 0]).item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError('Failed to align')
    return path[::-1]


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments, separator=' '):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def plot_alignments(
    trellis,
    segments,
    word_segments,
    waveform,
    separator: str = ' ',
    sr: int = 16000,
    figsize: tuple = (16, 9),
    plot_score_char: bool = False,
    plot_score_word: bool = True,
):
    """
    plot alignment using CTC alignment.

    Parameters
    ----------
    trellis: np.array
        usually `results['trellis']`.
    segments: list
        usually `results['segments']`.
    word_segments: list
        usually `results['word_segments']`.
    waveform: np.array
        input audio.
    separator: str, optional (default=' ')
        separator between words.
    sr: int, optional (default=16000)
    figsize: tuple, optional (default=(16, 9))
        figure size for matplotlib `figsize`.
    plot_score_char: bool, optional (default=False)
        plot score on top of character plots.
    plot_score_word: bool, optional (default=True)
        plot score on top of word plots.
    """

    trellis_with_path = trellis.copy()
    for i, seg in enumerate(segments):
        if seg.label != separator:
            trellis_with_path[seg.start + 1: seg.end + 1, i + 1] = float('nan')

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=figsize)

    ax1.imshow(trellis_with_path[1:, 1:].T, origin='lower')
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        ax1.axvline(word.start - 0.5)
        ax1.axvline(word.end - 0.5)

    for i, seg in enumerate(segments):
        if seg.label != separator:
            ax1.annotate(seg.label, (seg.start, i + 0.3))
            if plot_score_char:
                ax1.annotate(f'{seg.score:.2f}', (seg.start, i + 4), fontsize=8)

    # The original waveform
    ratio = waveform.shape[0] / (trellis.shape[0] - 1)
    ax2.plot(waveform)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color='red')
        if plot_score_word:
            ax2.annotate(f'{word.score:.2f}', (x0, 0.8))

    for seg in segments:
        if seg.label != separator:
            ax2.annotate(seg.label, (seg.start * ratio, 0.9))
    xticks = ax2.get_xticks()
    plt.xticks(xticks, xticks / sr)
    ax2.set_xlabel('time [second]')
    ax2.set_yticks([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(0, waveform.shape[0])
    plt.show()
