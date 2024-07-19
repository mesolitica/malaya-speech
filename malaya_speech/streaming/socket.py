import collections
from functools import partial
from datetime import datetime
import numpy as np
import logging


logger = logging.getLogger(__name__)


class Audio:
    def __init__(
        self,
        vad_model=None,
        segment_length: int = 480,
        num_padding_frames: int = 20,
        ratio: float = 0.25,
        mode_utterence: bool = True,
        enable_silent_timeout: bool = False,
        silent_timeout: float = 5.0,
        **kwargs,
    ):

        self.vad_model = vad_model
        self.segment_length = segment_length
        self.mode_utterence = mode_utterence
        self.enable_silent_timeout = enable_silent_timeout
        self.silent_timeout = silent_timeout
        self.queue = np.array([], np.float32)
        self.num_padding_frames = num_padding_frames
        self.ratio = ratio
        self.ring_buffer = collections.deque(maxlen=num_padding_frames)
        self.triggered = False
        self.i = 0
        self.last_silent = datetime.now()

    def vad_collector(self, array, **kwargs):
        """
        Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
        Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
        Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                    |---utterence---|        |---utterence---|
        """
        self.queue = np.concatenate([self.queue, array])
        chunks = []
        while True:
            t_ = self.queue[: self.segment_length]
            if len(t_) == self.segment_length:
                self.queue = self.queue[self.segment_length:]
                chunks.append(t_)
            else:
                break

        for chunk in chunks:
            frame = chunk

            if self.vad_model:
                try:
                    is_speech = self.vad_model(frame)
                    if isinstance(is_speech, dict):
                        is_speech = is_speech['vad']
                except Exception as e:
                    logger.debug(e)
                    is_speech = True
            else:
                is_speech = True

            logger.debug(is_speech)
            frame = (frame, self.i * self.segment_length)

            if self.mode_utterence:

                if not self.triggered:
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    if num_voiced > self.ratio * self.ring_buffer.maxlen:
                        self.triggered = True

                        self.last_silent = datetime.now()
                        for f, s in self.ring_buffer:
                            yield f

                        self.ring_buffer.clear()

                else:
                    self.last_silent = datetime.now()
                    yield frame

                    self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len(
                        [f for f, speech in self.ring_buffer if not speech]
                    )
                    if num_unvoiced > self.ratio * self.ring_buffer.maxlen:
                        self.triggered = False
                        yield None
                        self.ring_buffer.clear()

                if self.enable_silent_timeout and (
                        datetime.now() - self.last_silent).seconds >= self.silent_timeout:
                    self.last_silent = datetime.now()
                    yield None

            else:
                yield frame
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in self.ring_buffer if not speech]
                )
                if num_unvoiced > self.ratio * self.ring_buffer.maxlen:
                    yield None
                    self.ring_buffer.clear()

            self.i += 1
