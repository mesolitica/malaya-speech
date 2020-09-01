import webrtcvad
import threading, collections


class VAD:
    """Filter & segment audio with voice activity detection."""

    def __init__(
        self,
        input_rate: int = 16000,
        sample_rate: int = 16000,
        aggressiveness: int = 3,
        channels = 1,
        blocks_per_second = 50,
    ):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.input_rate = input_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocks_per_second = blocks_per_second
        self.block_size = int(self.sample_rate / float(self.blocks_per_second))
        self.block_size_input = int(
            self.input_rate / float(self.blocks_per_second)
        )
        self.frame_duration_ms = 1000 * self.block_size // self.sample_rate

    def frame_generator(self):
        yield

    def vad_collector(self, padding_ms = 300, ratio = 0.75, frames = None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen = num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech]
                )
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()
