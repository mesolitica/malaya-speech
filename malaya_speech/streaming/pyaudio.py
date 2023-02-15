# https://github.com/mozilla/DeepSpeech-examples/blob/r0.8/mic_vad_streaming/mic_vad_streaming.py

import collections
import queue
import os
import numpy as np
import logging
from datetime import datetime
from typing import List, Callable
from malaya_speech.utils.astype import float_to_int
from malaya_speech.streaming import stream as base_stream

logger = logging.getLogger(__name__)

pyaudio_available = False
try:
    import pyaudio
    pyaudio_available = True
except Exception as e:
    logger.warning(f'`pyaudio` is not available, `{__name__}` is not able to use.')


class Audio:
    def __init__(
        self,
        vad_model=None,
        callback=None,
        device=None,
        input_rate: int = 16000,
        sample_rate: int = 16000,
        segment_length: int = 320,
        channels: int = 1,
        stream_callback: Callable = None,
        **kwargs,
    ):

        self.vad_model = vad_model
        self.input_rate = input_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size_input = segment_length

        self.buffer_queue = queue.Queue()
        self.device = device
        self.format = pyaudio.paFloat32

        if callback is None:
            def callback(in_data): return self.buffer_queue.put(in_data)

        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        self.pa = pyaudio.PyAudio()
        kwargs = {
            'format': self.format,
            'channels': self.channels,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback if stream_callback is None else stream_callback,
        }

        self.chunk = None
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.sample_rate)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        return self.resample(
            data=self.buffer_queue.get(), input_rate=self.input_rate
        )

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def frame_generator(self):
        if self.input_rate == self.sample_rate:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def vad_collector(self, num_padding_frames=20, ratio=0.75):
        """
        Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
        Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
        Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                    |---utterence---|        |---utterence---|
        """
        frames = self.frame_generator()
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            frame = np.frombuffer(frame, np.float32)

            if self.vad_model:
                try:
                    is_speech = self.vad_model(frame)
                    if isinstance(is_speech, dict):
                        is_speech = is_speech['vad']
                except Exception as e:
                    logger.debug(e)
                    is_speech = False
            else:
                is_speech = True

            logger.debug(is_speech)

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


def stream(
    vad_model=None,
    asr_model=None,
    classification_model=None,
    sample_rate: int = 16000,
    segment_length: int = 2560,
    num_padding_frames: int = 20,
    ratio: float = 0.75,
    min_length: float = 0.1,
    max_length: float = 10.0,
    realtime_print: bool = True,
    **kwargs,
):
    """
    Stream an audio using pyaudio library.

    Parameters
    ----------
    vad_model: object, optional (default=None)
        vad model / pipeline.
    asr_model: object, optional (default=None)
        ASR model / pipeline, will transcribe each subsamples realtime.
    classification_model: object, optional (default=None)
        classification pipeline, will classify each subsamples realtime.
    device: None, optional (default=None)
        `device` parameter for pyaudio, check available devices from `sounddevice.query_devices()`.
    sample_rate: int, optional (default = 16000)
        output sample rate.
    segment_length: int, optional (default=2560)
        usually derived from asr_model.segment_length * asr_model.hop_length,
        size of audio chunks, actual size in term of second is `segment_length` / `sample_rate`.
    ratio: float, optional (default = 0.75)
        if 75% of the queue is positive, assumed it is a voice activity.
    min_length: float, optional (default=0.1)
        minimum length (second) to accept a subsample.
    max_length: float, optional (default=10.0)
        maximum length (second) to accept a subsample.
    realtime_print: bool, optional (default=True)
        Will print results for ASR.
    **kwargs: vector argument
        vector argument pass to malaya_speech.streaming.pyaudio.Audio interface.

    Returns
    -------
    result : List[dict]
    """
    if not pyaudio_available:
        raise ModuleNotFoundError(
            'pyaudio not installed. Please install it by `pip install pyaudio` and try again.'
        )

    return base_stream(
        audio_class=Audio,
        vad_model=vad_model,
        asr_model=asr_model,
        classification_model=classification_model,
        sample_rate=sample_rate,
        segment_length=segment_length,
        num_padding_frames=num_padding_frames,
        ratio=ratio,
        min_length=min_length,
        max_length=max_length,
        realtime_print=realtime_print,
        **kwargs,
    )
