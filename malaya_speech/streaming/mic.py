# https://github.com/mozilla/DeepSpeech-examples/blob/r0.8/mic_vad_streaming/mic_vad_streaming.py

import threading, collections, queue, os, os.path
import numpy as np
from malaya_speech.utils.vad import webrtc
import wave
from datetime import datetime

RATE_PROCESS = 16000
CHANNELS = 1
BLOCKS_PER_SECOND = 50


class Audio(webrtc.VAD):
    def __init__(
        self,
        callback = None,
        device = None,
        format = None,
        input_rate = RATE_PROCESS,
        sample_rate = RATE_PROCESS,
        channels = CHANNELS,
        blocks_per_second = BLOCKS_PER_SECOND,
    ):
        import pyaudio

        super().__init__(
            input_rate = input_rate,
            sample_rate = sample_rate,
            channels = channels,
            blocks_per_second = blocks_per_second,
        )

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


def record(
    model = None,
    device = None,
    filename: str = None,
    min_length: float = 1.5,
    spinner: bool = True,
):
    try:
        import pyaudio
    except:
        raise ValueError(
            'pyaudio not installed. Please install it by `pip install pyaudio` and try again.'
        )

    audio = Audio(device = device, format = pyaudio.paInt16)
    frames = audio.vad_collector()
    wav_data = bytearray()
    if spinner:
        from halo import Halo

        spinner = Halo(
            text = 'Listening (ctrl-C to exit) ...', spinner = 'line'
        )
    else:
        print('Listening (ctrl-C to exit) ... \n')

    try:
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

                if duration >= min_length:

                    if filename is None:
                        filename_temp = datetime.now().strftime(
                            'savewav_%Y-%m-%d_%H-%M-%S_%f.wav'
                        )
                    else:
                        filename_temp = filename

                    print(f'saved audio to {filename_temp}')
                    audio.write_wav(filename_temp, wav_data)
                    wav_data = bytearray()

    except KeyboardInterrupt:

        if filename is None:
            filename_temp = datetime.now().strftime(
                'savewav_%Y-%m-%d_%H-%M-%S_%f.wav'
            )
        else:
            filename_temp = filename

        print(f'saved audio to {filename_temp}')
        audio.write_wav(filename_temp, wav_data)

    if spinner:
        spinner.stop()

    audio.destroy()
