import numpy as np
from itertools import cycle, product
from malaya_speech.model.frame import Frame
from herpetologist import check_type
from typing import List, Tuple
from itertools import groupby


def get_ax(
    ax = None,
    xlim = (0, 1000),
    ylim = (0, 1),
    yaxis = False,
    time = True,
    **kwargs
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
    preds: List[Tuple[Frame, bool]],
    sample_rate: int = 16000,
    figsize: Tuple[int, int] = (15, 3),
    ax = None,
    **kwargs
):
    """
    Visualize signal given VAD labels. Green means got voice activity, while Red is not.

    Parameters
    -----------
    signal: list / np.array
    preds: List[Tuple[Frame, bool]]
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

    if ax is None:
        sns.set()
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1, 1, 1)
        plot = True
    else:
        min_timestamp = min([i[0].timestamp for i in preds])
        max_timestamp = max([i[0].timestamp + i[0].duration for i in preds])

        ax = get_ax(
            ax,
            xlim = (min_timestamp, max_timestamp),
            ylim = (np.min(signal), np.max(signal)),
            **kwargs
        )
        plot = False
    ax.plot([i / sample_rate for i in range(len(signal))], signal)
    for predictions in preds:
        color = 'g' if predictions[1] else 'r'
        p = predictions[0]
        ax.axvspan(
            p.timestamp, p.timestamp + p.duration, alpha = 0.5, color = color
        )
    if plot:
        plt.xlabel('Time (s)', size = 20)
        plt.ylabel('Amplitude', size = 20)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.show()


def plot_classification(
    preds,
    description,
    ax = None,
    fontsize_text = 14,
    x_text = 0.05,
    y_text = 0.2,
    ylim = (0.1, 0.9),
    figsize: Tuple[int, int] = (15, 3),
    **kwargs
):
    """
    Visualize probability / boolean.
    
    Parameters
    -----------
    preds: List[Tuple[Frame, label]]
    description: str
    ax: ax, optional (default = None)
    fontsize_text: int, optional (default = 14)
    x_text: float, optional (default = 0.05)
    y_text: float, optional (default = 0.2)
    """

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except:
        raise ValueError(
            'seaborn and matplotlib not installed. Please install it by `pip install matplotlib seaborn` and try again.'
        )

    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1, 1, 1)

    if isinstance(preds[0][1], float) or isinstance(preds[0][1], np.float32):
        hline = False
    else:
        hline = True

    min_timestamp = min([i[0].timestamp for i in preds])
    max_timestamp = max([i[0].timestamp + i[0].duration for i in preds])
    ax = get_ax(ax, xlim = (min_timestamp, max_timestamp), **kwargs)

    if hline:
        x = [i[1] for i in preds]
        labels = sorted(list(set(x)))
        styles = get_styles(len(labels))
        styles = {label: style for label, style in zip(labels, styles)}
        xs = [labels.index(i[1]) for i in preds]
        a = np.array(xs)
        std = (a - np.min(a)) / (np.max(a) - np.min(a))
        scaled = std * (ylim[1] - ylim[0]) + ylim[0]

        for i in range(len(preds)):
            linestyle, linewidth, color = styles[x[i]]
            ax.hlines(
                scaled[i],
                preds[i][0].timestamp,
                preds[i][0].timestamp + preds[i][0].duration,
                color,
                linewidth = linewidth,
                linestyle = linestyle,
                label = x[i],
            )
            ax.vlines(
                preds[i][0].timestamp,
                scaled[i] + 0.05,
                scaled[i] - 0.05,
                color,
                linewidth = 1,
                linestyle = 'solid',
            )
            ax.vlines(
                preds[i][0].timestamp + preds[i][0].duration,
                scaled[i] + 0.05,
                scaled[i] - 0.05,
                color,
                linewidth = 1,
                linestyle = 'solid',
            )
        H, L = ax.get_legend_handles_labels()

        HL = groupby(
            sorted(zip(H, L), key = lambda h_l: h_l[1]),
            key = lambda h_l: h_l[1],
        )
        H, L = zip(*list((next(h_l)[0], l) for l, h_l in HL))
        ax.legend(
            H,
            L,
            bbox_to_anchor = (0, 1),
            loc = 3,
            ncol = 5,
            borderaxespad = 0.0,
            frameon = False,
        )

    else:
        x = [i[0].timestamp for i in preds]
        y = [i[1] for i in preds]
        ax.plot(x, y)

    x = [i[0].timestamp for i in preds]

    ax.text(
        x[int(len(x) * x_text)], y_text, description, fontsize = fontsize_text
    )

    return ax
