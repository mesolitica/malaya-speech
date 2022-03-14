import numpy as np
from dataclasses import dataclass
from malaya_speech.utils.char import CTC_VOCAB as labels


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    from scipy.stats import betabinom

    x = np.arange(0, phoneme_count)
    mel_text_probs = []
    for i in range(1, mel_count + 1):
        a, b = scaling_factor * i, scaling_factor * (mel_count + 1 - i)
        mel_i_prob = betabinom(phoneme_count, a, b).pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1

    assert opt.sum(0).all()
    assert opt.sum(1).all()

    return opt


def binarize_attention(attn, in_len, out_len):
    b_size = attn.shape[0]
    attn_cpu = attn
    attn_out = np.zeros_like(attn)
    for ind in range(b_size):
        hard_attn = mas(attn_cpu[ind, 0, : out_len[ind], : in_len[ind]])
        attn_out[ind, 0, : out_len[ind], : in_len[ind]] = hard_attn
    return attn_out


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


def put_comma(alignment, min_threshold: float = 0.5):
    """
    Put comma in alignment from force alignment model.

    Parameters
    -----------
    alignment: List[Dict[text, start, end]]
    min_threshold: float, optional (default=0.5)
        minimum threshold in term of seconds to assume a comma.

    Returns
    --------
    result: List[str]
    """
    r = []
    for no, row in enumerate(alignment):
        if no > 0:
            if alignment[no]['start'] - alignment[no-1]['end'] >= min_threshold:
                r.append(',')

        r.append(row['text'])
    r.append('.')
    return r


def plot_alignments(
    alignment,
    subs_alignment,
    words_alignment,
    waveform,
    separator: str = ' ',
    sample_rate: int = 16000,
    figsize: tuple = (16, 9),
    plot_score_char: bool = False,
    plot_score_word: bool = True,
):
    """
    plot alignment.

    Parameters
    ----------
    alignment: np.array
        usually `alignment` output.
    subs_alignment: list
        usually `chars_alignment` or `subwords_alignment` output.
    words_alignment: list
        usually `words_alignment` output.
    waveform: np.array
        input audio.
    separator: str, optional (default=' ')
        separator between words, only useful if `subs_alignment` is character based.
    sample_rate: int, optional (default=16000)
    figsize: tuple, optional (default=(16, 9))
        figure size for matplotlib `figsize`.
    plot_score_char: bool, optional (default=False)
        plot score on top of character plots.
    plot_score_word: bool, optional (default=True)
        plot score on top of word plots.
    """

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except BaseException:
        raise ValueError(
            'seaborn and matplotlib not installed. Please install it by `pip install matplotlib seaborn` and try again.'
        )

    trellis_with_path = alignment.copy()
    if trellis_with_path.shape[1] == len(subs_alignment):
        trellis_with_path = np.concatenate([trellis_with_path,
                                            np.zeros((trellis_with_path.shape[0], 2))],
                                           axis=1)

    for i, seg in enumerate(subs_alignment):
        if seg['text'] != separator:
            trellis_with_path[seg['start_t'] + 1: seg['end_t'] + 1, i + 1] = float('nan')

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=figsize)

    ax1.imshow(trellis_with_path[1:, 1:].T, origin='lower')
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in words_alignment:
        ax1.axvline(word['start_t'] - 0.5)
        ax1.axvline(word['end_t'] - 0.5)

    for i, seg in enumerate(subs_alignment):
        if seg['text'] != separator:
            ax1.annotate(seg['text'], (seg['start_t'], i + 0.3))
            if plot_score_char:
                ax1.annotate(f"{seg['score']:.2f}", (seg['start_t'], i + 4), fontsize=8)

    ratio = waveform.shape[0] / (alignment.shape[0] - 1)
    ax2.plot(waveform)
    for word in words_alignment:
        x0 = ratio * word['start_t']
        x1 = ratio * word['end_t']
        ax2.axvspan(x0, x1, alpha=0.1, color='red')
        if plot_score_word:
            ax2.annotate(f"{word['score']:.2f}", (x0, 0.8))

    for seg in subs_alignment:
        if seg['text'] != separator:
            ax2.annotate(seg['text'], (seg['start_t'] * ratio, 0.9))

    xticks = ax2.get_xticks()
    plt.xticks(xticks, xticks / sample_rate)
    ax2.set_xlabel('time (second)')
    ax2.set_yticks([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(0, waveform.shape[0])
    plt.show()
