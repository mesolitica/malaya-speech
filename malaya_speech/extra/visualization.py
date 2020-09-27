from itertools import cycle, product
from malaya_speech.model.frame import FRAME
from herpetologist import check_type
from typing import List, Tuple


def get_ax(
    ax = None, xlim = (0, 1000), ylim = (0, 1), yaxis = False, time = True
):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except:
        raise ValueError(
            'seaborn and matplotlib not installed. Please install it by `pip install matplotlib seaborn` and try again.'
        )

    if ax is None:
        ax = plt.gca()
    ax.set_xlim(xlim)
    if time:
        ax.set_xlabel('Time')
    else:
        ax.set_xticklabels([])
    ax.set_ylim(ylim)
    ax.axes.get_yaxis().set_visible(yaxis)
    return ax


def get_styles(size):
    try:
        from matplotlib.cm import get_cmap
    except:
        raise ValueError(
            'matplotlib not installed. Please install it by `pip install matplotlib` and try again.'
        )

    linewidth = [3, 1]
    linestyle = ['solid', 'dashed', 'dotted']

    cm = get_cmap('Set1')
    colors = [cm(1.0 * i / 8) for i in range(9)]

    style_generator = cycle(product(linestyle, linewidth, colors))
    styles = [next(style_generator) for _ in range(size)]
    return styles


def visualize_vad(
    signal,
    preds: List[Tuple[FRAME, bool]],
    sample_rate: int = 16000,
    figsize: Tuple[int, int] = (15, 7),
):
    """
    Visualize signal given VAD labels. Green means got voice activity, while Red is not.

    Parameters
    -----------
    signal: list / np.array
    preds: List[Tuple[FRAME, bool]]
    sample_rate: int, optional (default=16000)
    figsize: Tuple[int, int], optional (default=(15, 7))
        matplotlib figure size.

    """

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except:
        raise ValueError(
            'seaborn and matplotlib not installed. Please install it by `pip install matplotlib seaborn` and try again.'
        )

    fig = plt.figure(figsize = figsize)
    sns.set()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([i / sample_rate for i in range(len(signal))], signal)
    for predictions in preds:
        color = 'g' if predictions[1] else 'r'
        p = predictions[0]
        ax.axvspan(
            p.timestamp, p.timestamp + p.duration, alpha = 0.5, color = color
        )
    plt.xlabel('Time (s)', size = 20)
    plt.ylabel('Amplitude', size = 20)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.show()
