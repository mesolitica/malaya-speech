from malaya_speech.model.interface import FRAME
from herpetologist import check_type
from typing import List, Tuple


def visualize_vad(
    signal,
    preds: List[Tuple[FRAME, bool]],
    sample_rate = 16000,
    figsize = (15, 7),
):
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
