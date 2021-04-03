# https://github.com/mozilla/DeepSpeech-examples/blob/r0.8/mic_vad_streaming/mic_vad_streaming.py

import threading, collections, queue, os, os.path
import numpy as np
import wave
from datetime import datetime
from malaya_speech.utils.validator import check_pipeline
from herpetologist import check_type


class Audio:
    @check_type
    def __init__(
        self,
        vad,
        callback = None,
        device = None,
        format = None,
        input_rate: int = 16000,
        sample_rate: int = 16000,
        blocks_per_second: int = 50,
        channels: int = 1,
    ):

        import pyaudio

        self.vad = vad
        self.input_rate = input_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocks_per_second = blocks_per_second
        self.block_size = int(self.sample_rate / float(self.blocks_per_second))
        self.block_size_input = int(
            self.input_rate / float(self.blocks_per_second)
        )
        self.frame_duration_ms = 1000 * self.block_size // self.sample_rate

        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None:
            callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.format = format

        self.pa = pyaudio.PyAudio()
        kwargs = {
            'format': format,
            'channels': self.channels,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        data16 = np.fromstring(string = data, dtype = np.int16)
        resample_size = int(len(data16) / self.input_rate * self.sample_rate)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype = np.int16)
        return resample16.tostring()

    def read_resampled(self):
        return self.resample(
            data = self.buffer_queue.get(), input_rate = self.input_rate
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

    def write_wav(self, filename, data):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

    def vad_collector(self, padding_ms = 300, ratio = 0.75):
        """
        Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
        Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
        Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                    |---utterence---|        |---utterence---|
        """
        frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen = num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad(frame)
            if isinstance(is_speech, dict):
                is_speech = is_speech['vad']

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


def record(
    vad,
    asr_model = None,
    classification_model = None,
    device = None,
    input_rate: int = 16000,
    sample_rate: int = 16000,
    blocks_per_second: int = 50,
    padding_ms: int = 300,
    ratio: float = 0.75,
    min_length: float = 0.1,
    filename: str = None,
    spinner: bool = False,
):
    """
    Record an audio using pyaudio library. This record interface required a VAD model.

    Parameters
    ----------
    vad: object
        vad model / pipeline.
    asr_model: object
        ASR model / pipeline, will transcribe each subsamples realtime.
    classification_model: object
        classification pipeline, will classify each subsamples realtime.
    device: None
        `device` parameter for pyaudio, check available devices from `sounddevice.query_devices()`.
    input_rate: int, optional (default = 16000)
        sample rate from input device, this will auto resampling.
    sample_rate: int, optional (default = 16000)
        output sample rate.
    blocks_per_second: int, optional (default = 50)
        size of frame returned from pyaudio, frame size = sample rate / (blocks_per_second / 2).
        50 is good for WebRTC, 30 or less is good for Malaya Speech VAD.
    padding_ms: int, optional (default = 300)
        size of queue to store frames, size = padding_ms // (1000 * blocks_per_second // sample_rate)
    ratio: float, optional (default = 0.75)
        if 75% of the queue is positive, assumed it is a voice activity.
    min_length: float, optional (default=0.1)
        minimum length (s) to accept a subsample.
    filename: str, optional (default=None)
        if None, will auto generate name based on timestamp.
    spinner: bool, optional (default=False)
        if True, will use spinner object from halo library.


    Returns
    -------
    result : [filename, samples]
    """

    try:
        import pyaudio
    except:
        raise ModuleNotFoundError(
            'pyaudio not installed. Please install it by `pip install pyaudio` and try again.'
        )

    check_pipeline(vad, 'vad', 'vad')
    if asr_model:
        check_pipeline(asr_model, 'speech-to-text', 'asr_model')
    if classification_model:
        check_pipeline(
            classification_model, 'classification', 'classification_model'
        )

    audio = Audio(
        vad,
        device = device,
        input_rate = input_rate,
        sample_rate = sample_rate,
        format = pyaudio.paInt16,
        blocks_per_second = blocks_per_second,
    )
    frames = audio.vad_collector(padding_ms = padding_ms, ratio = ratio)

    if spinner:
        try:
            from halo import Halo
        except:
            raise ModuleNotFoundError(
                'halo not installed. Please install it by `pip install halo` and try again, or simply set `spinner=False`.'
            )

        spinner = Halo(
            text = 'Listening (ctrl-C to stop recording) ...', spinner = 'line'
        )
    else:
        print('Listening (ctrl-C to stop recording) ... \n')

    results = []
    wav_data = bytearray()

    try:
        count = 0
        for frame in frames:
            if frame is not None:
                if spinner:
                    spinner.start()
                wav_data.extend(frame)
            else:
                if spinner:
                    spinner.stop()

                buffered = np.frombuffer(wav_data, np.int16)
                duration = buffered.shape[0] / audio.input_rate

                wav_data = [wav_data]

                if duration >= min_length:
                    if asr_model:
                        t = asr_model(wav_data[0])
                        if isinstance(t, dict):
                            t = t['speech-to-text']
                        print(f'Sample {count} {datetime.now()}: {t}')
                        wav_data.append(t)
                    if classification_model:
                        t = classification_model(wav_data[0])
                        if isinstance(t, dict):
                            t = t['classification']
                        print(f'Sample {count} {datetime.now()}: {t}')
                        wav_data.append(t)

                    results.append(wav_data)
                    wav_data = bytearray()
                    count += 1

    except KeyboardInterrupt:

        if filename is None:
            filename_temp = datetime.now().strftime(
                'savewav_%Y-%m-%d_%H-%M-%S_%f.wav'
            )
        else:
            filename_temp = filename

        print(f'saved audio to {filename_temp}')

        bytes_array = [r[0] for r in results]
        audio.write_wav(filename_temp, b''.join(bytes_array))

    except Exception as e:
        raise e

    if spinner:
        spinner.stop()

    audio.destroy()
    return filename_temp, results
