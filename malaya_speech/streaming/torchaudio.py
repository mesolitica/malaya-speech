"""
https://pytorch.org/audio/stable/tutorials/online_asr_tutorial.html
"""

from malaya_speech.utils.torch_featurization import StreamReader, torchaudio_available
from malaya_speech.torch_model.torchaudio import Conformer
import torch
import logging

logger = logging.getLogger(__name__)


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


def stream(src, model, beam_width: int = 10):
    """
    Parameters
    -----------
    src: str
        Supported `src` for `torchaudio.io.StreamReader`
        Read more at https://pytorch.org/audio/stable/tutorials/streamreader_basic_tutorial.html#sphx-glr-tutorials-streamreader-basic-tutorial-py
        or https://pytorch.org/audio/stable/tutorials/streamreader_advanced_tutorial.html#sphx-glr-tutorials-streamreader-advanced-tutorial-py
    model: Callable
        must `malaya_speech.torch_model.conformer.Conformer` class.
    beam_width: int, optional (default=10)
        width for beam decoding.
    """

    if not isinstance(model, Conformer):
        raise ValueError('model must a conformer.')

    sample_rate = model.sample_rate
    segment_length = model.segment_length * model.hop_length
    context_length = model.right_context_length * model.hop_length

    streamer = StreamReader(src=src)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=model.sample_rate)

    logger.info(streamer.get_src_stream_info(0))

    cacher = ContextCacher(segment_length, context_length)
    stream_iterator = streamer.stream()

    @torch.inference_mode()
    def run_inference(num_iter=200, state=None, hypothesis=None):
        chunks = []
        for i, (chunk,) in enumerate(stream_iterator, start=1):
            segment = cacher(chunk[:, 0])
            features, length = model.feature_extractor(segment)
            hypos, state = model.decoder.infer(features, length, beam_width, state=state, hypothesis=hypothesis)
            hypothesis = hypos[0]
            transcript = model.token_processor(hypothesis[0], lstrip=False)
            print(transcript, end='', flush=True)

            chunks.append(chunk)
            if i == num_iter:
                break

    run_inference()
