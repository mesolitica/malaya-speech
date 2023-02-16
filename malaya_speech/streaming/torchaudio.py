"""
https://pytorch.org/audio/stable/tutorials/online_asr_tutorial.html
"""
import collections
from datetime import datetime
from malaya_speech.utils.validator import check_pipeline
from malaya_speech.utils.torch_featurization import StreamReader, torchaudio_available
from malaya_speech.torch_model.torchaudio import Conformer
from malaya_speech.streaming import stream as base_stream
from functools import partial
import torch
import logging

logger = logging.getLogger(__name__)

if StreamReader is None:
    logger.warning(f'`torchaudio.io.StreamReader` is not available, `{__name__}` is not able to use.')


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length:]
        return chunk_with_context


def _base_stream(
    src,
    format=None,
    option=None,
    buffer_size: int = 4096,
    sample_rate: int = 16000,
    segment_length: int = 2560,
):

    if StreamReader is None:
        raise ValueError('`torchaudio.io.StreamReader is not available, please make sure your ffmpeg installed properly.')

    streamer = StreamReader(src=src, format=format, option=option, buffer_size=buffer_size)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

    logger.info(streamer.get_src_stream_info(0))
    stream_iterator = streamer.stream()
    return streamer.stream()


class Audio:
    def __init__(
        self,
        src,
        vad_model=None,
        format=None,
        option=None,
        buffer_size: int = 4096,
        sample_rate: int = 16000,
        segment_length: int = 2560,
        **kwargs,
    ):
        self.vad_model = vad_model
        self.stream_iterator = _base_stream(
            src=src,
            format=format,
            option=option,
            buffer_size=buffer_size,
            sample_rate=sample_rate,
            segment_length=segment_length,
        )
        self.segment_length = segment_length

    def destroy(self):
        pass

    def vad_collector(self, num_padding_frames=20, ratio=0.75):
        """
        Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
        Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
        Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                    |---utterence---|        |---utterence---|
        """
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for i, (chunk,) in enumerate(self.stream_iterator, start=1):
            frame = chunk[:, 0].numpy()
            if len(frame) != self.segment_length:
                continue

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
    src,
    vad_model=None,
    asr_model=None,
    classification_model=None,
    format=None,
    option=None,
    buffer_size: int = 4096,
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
    Stream an audio using torchaudio library.

    Parameters
    ----------
    vad_model: object, optional (default=None)
        vad model / pipeline.
    asr_model: object, optional (default=None)
        ASR model / pipeline, will transcribe each subsamples realtime.
    classification_model: object, optional (default=None)
        classification pipeline, will classify each subsamples realtime.
    format: str, optional (default=None)
        Supported `format` for `torchaudio.io.StreamReader`,
        https://pytorch.org/audio/stable/generated/torchaudio.io.StreamReader.html#torchaudio.io.StreamReader
    option: dict, optional (default=None)
        Supported `option` for `torchaudio.io.StreamReader`,
        https://pytorch.org/audio/stable/generated/torchaudio.io.StreamReader.html#torchaudio.io.StreamReader
    buffer_size: int, optional (default=4096)
        Supported `buffer_size` for `torchaudio.io.StreamReader`, buffer size in byte. Used only when src is file-like object,
        https://pytorch.org/audio/stable/generated/torchaudio.io.StreamReader.html#torchaudio.io.StreamReader
    sample_rate: int, optional (default = 16000)
        output sample rate.
    segment_length: int, optional (default=2560)
        usually derived from asr_model.segment_length * asr_model.hop_length,
        size of audio chunks, actual size in term of second is `segment_length` / `sample_rate`.
    num_padding_frames: int, optional (default=20)
        size of acceptable padding frames for queue.
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

    return base_stream(
        audio_class=partial(Audio, src=src, format=format, option=option, buffer_size=buffer_size),
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


def stream_rnnt(
    src,
    asr_model=None,
    classification_model=None,
    format=None,
    option=None,
    beam_width: int = 10,
    buffer_size: int = 4096,
    sample_rate: int = 16000,
    segment_length: int = 2560,
    context_length: int = 640,
    realtime_print: bool = True,
    **kwargs,
):
    """
    Parameters
    -----------
    src: str
        Supported `src` for `torchaudio.io.StreamReader`
        Read more at https://pytorch.org/audio/stable/tutorials/streamreader_basic_tutorial.html#sphx-glr-tutorials-streamreader-basic-tutorial-py
        or https://pytorch.org/audio/stable/tutorials/streamreader_advanced_tutorial.html#sphx-glr-tutorials-streamreader-advanced-tutorial-py
    asr_model: object, optional (default=None)
        ASR model / pipeline, will transcribe each subsamples realtime.
        must be an object of `malaya_speech.torch_model.torchaudio.Conformer`.
    classification_model: object, optional (default=None)
        classification pipeline, will classify each subsamples realtime.
    format: str, optional (default=None)
        Supported `format` for `torchaudio.io.StreamReader`,
        https://pytorch.org/audio/stable/generated/torchaudio.io.StreamReader.html#torchaudio.io.StreamReader
    option: dict, optional (default=None)
        Supported `option` for `torchaudio.io.StreamReader`,
        https://pytorch.org/audio/stable/generated/torchaudio.io.StreamReader.html#torchaudio.io.StreamReader
    buffer_size: int, optional (default=4096)
        Supported `buffer_size` for `torchaudio.io.StreamReader`, buffer size in byte. Used only when src is file-like object,
        https://pytorch.org/audio/stable/generated/torchaudio.io.StreamReader.html#torchaudio.io.StreamReader
    sample_rate: int, optional (default=16000)
        sample rate from input device, this will auto resampling.
    segment_length: int, optional (default=2560)
        usually derived from asr_model.segment_length * asr_model.hop_length,
        size of audio chunks, actual size in term of second is `segment_length` / `sample_rate`.
    context_length: int, optional (default=640)
        usually derived from asr_model.right_context_length * asr_model.hop_length,
        size of append context chunks, only useful for streaming RNNT.
    beam_width: int, optional (default=10)
        width for beam decoding.
    realtime_print: bool, optional (default=True)
        Will print results for ASR.
    """

    if not isinstance(asr_model, Conformer):
        raise ValueError('`asr_model` only support Enformer RNNT.')

    if not getattr(asr_model, 'rnnt_streaming', False):
        raise ValueError('`asr_model` only support Enformer RNNT.')

    if classification_model:
        check_pipeline(
            classification_model, 'classification', 'classification_model'
        )

    if asr_model.feature_extractor.pad:
        asr_model.feature_extractor.pad = False

    stream_iterator = _base_stream(
        src=src,
        format=format,
        option=option,
        buffer_size=buffer_size,
        sample_rate=sample_rate,
    )

    cacher = ContextCacher(segment_length, context_length)

    @torch.inference_mode()
    def run_inference(state=None, hypothesis=None):
        results = []
        try:
            for i, (chunk,) in enumerate(stream_iterator, start=1):

                audio = chunk[:, 0]
                wav_data = {
                    'wav_data': audio.numpy(),
                    'timestamp': datetime.now(),
                }

                segment = cacher(audio)
                features, length = asr_model.feature_extractor(segment)
                hypos, state = asr_model.decoder.infer(features, length, beam_width, state=state, hypothesis=hypothesis)
                hypothesis = hypos[0]
                transcript = asr_model.tokenizer(hypothesis[0], lstrip=False)

                wav_data['asr_model'] = transcript

                if len(transcript.strip()) and classification_model:
                    t_ = classification_model(wav_data['wav_data'])
                    if isinstance(t_, dict):
                        t_ = t_['classification']

                    wav_data['classification_model'] = t_

                if realtime_print:
                    print(transcript, end='', flush=True)

                results.append(wav_data)

        except KeyboardInterrupt:
            pass

        except Exception as e:
            raise e

        return results

    return run_inference()
