import numpy as np
from itertools import groupby
from malaya_speech.model.frame import Segment


class Annotation:
    def __init__(self, uri: str = None):
        try:
            from sortedcontainers import SortedDict
        except:
            raise ValueError(
                'sortedcontainers not installed. Please install it by `pip install sortedcontainers` and try again.'
            )
        self._uri = uri
        self._tracks = SortedDict()

        self._labels = {}
        self._labelNeedsUpdate = {}

        self._timeline = None
        self._timelineNeedsUpdate: bool = True

    @property
    def max(self):
        k = list(self._tracks.keys())
        return k[-1].end

    @property
    def min(self):
        k = list(self._tracks.keys())
        return k[0].start

    @property
    def labels(self):
        l = list(set([i[2] for i in self.itertracks()]))
        return sorted(l)

    def itertracks(self, yield_label = True):
        for segment, tracks in self._tracks.items():
            for track, lbl in sorted(
                tracks.items(), key = lambda tl: (str(tl[0]), str(tl[1]))
            ):
                if yield_label:
                    yield segment, track, lbl
                else:
                    yield segment, track

    def __getitem__(self, key):
        if isinstance(key, Segment):
            key = (key, '_')

        return self._tracks[key[0]][key[1]]

    def __setitem__(self, key, label):
        if isinstance(key, Segment):
            key = (key, '_')

        segment, track = key
        if not segment:
            return

        if segment not in self._tracks:
            self._tracks[segment] = {}
            self._timelineNeedsUpdate = True

        if track in self._tracks[segment]:
            old_label = self._tracks[segment][track]
            self._labelNeedsUpdate[old_label] = True

        self._tracks[segment][track] = label
        self._labelNeedsUpdate[label] = True

    def crop(
        self,
        from_t: float = None,
        to_t: float = None,
        mode: str = 'intersection',
    ):

        """
        Crop sample by time.

        Parameters
        ----------
        from_t: float, optional (default=None)
            if None, will take self.min
        to_t: float, optional (default=None)
            if None, will take self.max
        mode: str, optional (default='intersection')
            crop mode supported. Allowed values:

            * ``'intersection'`` - sampling with crop if middle of the track.
            * ``'strict'`` - sampling with strictly method, will not crop the track.
            * ``'loose'`` - sampling with loose method, will take entire track.

        Returns
        -------
        result : malaya_speech.model.annotation.Annotation class
        """

        mode = mode.lower()
        if mode not in ['intersection', 'strict', 'loose']:
            raise ValueError(
                'mode only supported [`intersection`, `strict`, `loose`]'
            )

        if not from_t:
            from_t = 0
        if not to_t:
            to_t = self.max

        cropped = Annotation(self._uri)

        for segment, i, speaker in self.itertracks():
            if (
                mode == 'strict'
                and segment.start >= from_t
                and segment.end <= to_t
            ):
                cropped[segment, i] = speaker

            elif mode in ['loose', 'intersection'] and (
                (
                    segment.start <= from_t <= segment.end
                    or segment.start <= to_t <= segment.end
                )
                or (segment.start >= from_t and segment.end <= to_t)
            ):
                if (
                    mode == 'intersection'
                    and segment.start <= from_t <= segment.end
                ):
                    segment = Segment(from_t, segment.end)
                if (
                    mode == 'intersection'
                    and segment.start <= to_t <= segment.end
                ):
                    segment = Segment(segment.start, to_t)
                cropped[segment, i] = speaker

        return cropped

    def plot(self, ax = None):

        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except:
            raise ValueError(
                'seaborn and matplotlib not installed. Please install it by `pip install matplotlib seaborn` and try again.'
            )

        from malaya_speech.extra.visualization import get_ax, get_styles

        def get_y(segments):
            up_to = [-np.inf]
            y = []

            for segment in segments:
                found = False
                for i, u in enumerate(up_to):
                    if segment.start >= u:
                        found = True
                        y.append(i)
                        up_to[i] = segment.end
                        break
                if not found:
                    y.append(len(up_to))
                    up_to.append(segment.end)

            y = 1.0 - 1.0 / (len(up_to) + 1) * (1 + np.array(y))
            return y

        def draw_segment(
            ax, segment, y, styles, label = None, boundaries = True
        ):

            if not segment:
                return

            linestyle, linewidth, color = styles[label]
            ax.hlines(
                y,
                segment.start,
                segment.end,
                color,
                linewidth = linewidth,
                linestyle = linestyle,
                label = label,
            )
            if boundaries:
                ax.vlines(
                    segment.start,
                    y + 0.05,
                    y - 0.05,
                    color,
                    linewidth = 1,
                    linestyle = 'solid',
                )
                ax.vlines(
                    segment.end,
                    y + 0.05,
                    y - 0.05,
                    color,
                    linewidth = 1,
                    linestyle = 'solid',
                )

            if label is None:
                return

        time = True
        styles = get_styles(len(self.labels))
        styles = {label: style for label, style in zip(self.labels, styles)}

        if ax is None:
            figsize = plt.rcParams['figure.figsize']
            plt.rcParams['figure.figsize'] = (20, 2)
            fig, ax = plt.subplots()

        labels = self.labels
        xlim = (self.min, self.max)
        segments = [s for s, _ in self.itertracks(yield_label = False)]
        ax = get_ax(ax = ax, xlim = xlim, time = time)
        for (segment, track, label), y in zip(
            self.itertracks(), get_y(segments)
        ):
            draw_segment(ax, segment, y, styles, label = label)

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
        return ax
