import numpy as np
from dataclasses import dataclass

SEGMENT_PRECISION = 1e-6


class Frame:
    def __init__(self, array, timestamp, duration):
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        self.array = array
        self.timestamp = timestamp
        self.duration = duration


@dataclass(frozen = True, order = True)
class Segment:
    start: float = 0.0
    end: float = 0.0

    def __bool__(self):
        return bool((self.end - self.start) > SEGMENT_PRECISION)

    @property
    def duration(self) -> float:
        """
        Segment duration (read-only)
        """
        return self.end - self.start if self else 0.0

    @property
    def middle(self) -> float:
        """Segment mid-time (read-only)"""
        return 0.5 * (self.start + self.end)

    def __contains__(self, other: 'Segment'):
        """Inclusion
        >>> segment = Segment(start=0, end=10)
        >>> Segment(start=3, end=10) in segment:
        True
        >>> Segment(start=5, end=15) in segment:
        False
        """
        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        """
        Intersection
        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment & other_segment
        <Segment(5, 10)>
        Note
        ----
        When the intersection is empty, an empty segment is returned:
        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> intersection = segment & other_segment
        >>> if not intersection:
        ...    # intersection is empty.
        """
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Segment(start = start, end = end)

    def intersects(self, other: 'Segment') -> bool:
        """
        Check whether two segments intersect each other
        Parameters
        ----------
        other : Segment
            Other segment
        Returns
        -------
        intersect : bool
            True if segments intersect, False otherwise
        """

        return (
            (
                self.start < other.start
                and other.start < self.end - SEGMENT_PRECISION
            )
            or (
                self.start > other.start
                and self.start < other.end - SEGMENT_PRECISION
            )
            or (self.start == other.start)
        )

    def overlaps(self, t: float):
        """
        Check if segment overlaps a given time
        Parameters
        ----------
        t : float
            Time, in seconds.
        Returns
        -------
        overlap: bool
            True if segment overlaps time t, False otherwise.
        """
        return self.start <= t and self.end >= t

    def __or__(self, other):
        """
        Union
        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment | other_segment
        <Segment(0, 15)>
        Note
        ----
        When a gap exists between the segment, their union covers the gap as well:
        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment | other_segment
        <Segment(0, 20)
        """

        if not self:
            return other
        if not other:
            return self
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return Segment(start = start, end = end)

    def __xor__(self, other):
        """
        Gap
        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment ^ other_segment
        <Segment(10, 15)
        Note
        ----
        The gap between a segment and an empty segment is not defined.
        >>> segment = Segment(0, 10)
        >>> empty_segment = Segment(11, 11)
        >>> segment ^ empty_segment
        ValueError: The gap between a segment and an empty segment is not defined.
        """

        if (not self) or (not other):
            raise ValueError(
                'The gap between a segment and an empty segment '
                'is not defined.'
            )

        start = min(self.end, other.end)
        end = max(self.start, other.start)
        return Segment(start = start, end = end)

    def _str_helper(self, seconds: float):
        from datetime import timedelta

        negative = seconds < 0
        seconds = abs(seconds)
        td = timedelta(seconds = seconds)
        seconds = td.seconds + 86400 * td.days
        microseconds = td.microseconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '%s%02d:%02d:%02d.%03d' % (
            '-' if negative else ' ',
            hours,
            minutes,
            seconds,
            microseconds / 1000,
        )

    def __str__(self):
        """
        Human-readable representation
        >>> print(Segment(1337, 1337 + 0.42))
        [ 00:22:17.000 -->  00:22:17.420]
        Note
        ----
        Empty segments are printed as "[]"
        """
        return '<Segment(%g, %g)>' % (self.start, self.end)

    def __repr__(self):
        """
        Computer-readable representation
        >>> Segment(1337, 1337 + 0.42)
        <Segment(1337, 1337.42)>
        """
        return '<Segment(%g, %g)>' % (self.start, self.end)
