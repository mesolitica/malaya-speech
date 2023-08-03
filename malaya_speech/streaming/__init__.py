from malaya_speech.utils.validator import check_pipeline
from malaya_speech.model.frame import Frame
from scipy.io.wavfile import write
from datetime import datetime
from typing import Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


def stream(
    audio_class=None,
    vad_model=None,
    postprocessing_model=None,
    postfilter_model=None,
    asr_model=None,
    classification_model=None,
    sample_rate: int = 16000,
    segment_length: int = 2560,
    num_padding_frames: int = 20,
    ratio: float = 0.75,
    min_length: float = 0.1,
    max_length: float = 10.0,
    streaming_max_length: float = None,
    starting_timestamp: float = None,
    filename: str = None,
    realtime_print: bool = True,
    return_as_frame: bool = False,
    use_tqdm: bool = False,
    callback_pred: Callable = None,
    frames=None,
    **kwargs,
):

    if vad_model:
        check_pipeline(vad_model, 'vad', 'vad')
    if postprocessing_model:
        check_pipeline(postprocessing_model, 'postprocessing', 'postprocessing_model')
    if postfilter_model:
        check_pipeline(postfilter_model, 'postfilter', 'postfilter_model')
    if asr_model:
        check_pipeline(asr_model, 'speech-to-text', 'asr_model')
    if classification_model:
        check_pipeline(
            classification_model, 'classification', 'classification_model'
        )

    if audio_class is not None:
        audio = audio_class(
            vad_model=vad_model,
            sample_rate=sample_rate,
            segment_length=segment_length,
            **kwargs,
        )
        frames = audio.vad_collector(num_padding_frames=num_padding_frames, ratio=ratio)
    else:
        if frames is None:
            raise ValueError(
                'if `audio_class` parameter is None, `frames` parameter cannot be None')

    results = []
    indices = []
    wav_data = np.array([], dtype=np.float32)
    length = 0
    total_length = 0
    count = 0

    if use_tqdm:
        try:
            from tqdm import tqdm

            frames = tqdm(frames)
        except Exception as e:
            raise ValueError('tqdm is not available, please install it and try again.')

    def pred():

        nonlocal wav_data, indices

        now = indices[0] / sample_rate
        data_dict = {
            'wav_data': wav_data,
            'start': now,
        }
        t = ''

        if postprocessing_model:
            wav_data_ = postprocessing_model(wav_data)
            if isinstance(wav_data_, dict):
                logger.debug(wav_data_)
                wav_data_ = wav_data_['postprocessing']

            wav_data = wav_data_
            data_dict['wav_data'] = wav_data

        cont = True

        if postfilter_model:
            t_ = postfilter_model(wav_data)
            if isinstance(t_, dict):
                logger.debug(t_)
                t_ = t_['postfilter']
            cont = t_

        if cont:
            if asr_model:
                t_ = asr_model(wav_data)
                if isinstance(t_, dict):
                    logger.debug(t_)
                    t_ = t_['speech-to-text']

                data_dict['asr_model'] = t_
                logger.info(f'Sample asr_model {count} {now}: {t_}')

                t += str(t_) + ' '

            if classification_model:
                t_ = classification_model(wav_data)
                if isinstance(t_, dict):
                    logger.debug(t_)
                    t_ = t_['classification']

                data_dict['classification_model'] = t_
                logger.info(f'Sample classification_model {count} {now}: {t_}')

                t += f'({t_})' + ' '

            if realtime_print and len(t):
                print(t, end='', flush=True)

            data_dict['end'] = (indices[-1] + segment_length) / sample_rate

            if callback_pred is not None:
                callback_pred(data_dict)

            else:
                results.append(data_dict)

    try:
        for frame in frames:

            if frame is not None:
                frame, index = frame
                length += frame.shape[0] / sample_rate
                total_length += frame.shape[0] / sample_rate

                if starting_timestamp is not None and total_length < starting_timestamp:
                    continue

                wav_data = np.concatenate([wav_data, frame])
                indices.append(index)

            if frame is None and len(indices) and (length >= min_length or length >= max_length):

                pred()
                wav_data = np.array([], dtype=np.float32)
                indices = []
                length = 0
                count += 1

            if frame is None and streaming_max_length is not None:
                if starting_timestamp is not None:
                    maxlen = streaming_max_length + starting_timestamp
                else:
                    maxlen = streaming_max_length
                if total_length >= maxlen:
                    break

    except KeyboardInterrupt:
        pass

    except Exception as e:
        raise e

    if length > 0:
        pred()
        count += 1

    if filename is not None:
        logger.info(f'saved audio to {filename}')
        write(filename, np.concatenate([r[0] for r in results]))

    if audio_class is not None:
        audio.destroy()

    if return_as_frame:
        for i in range(len(results)):
            wav_data = results[i]['wav_data']
            duration = len(wav_data) / sample_rate
            frame = Frame(wav_data, results[i]['start'], duration)
            results[i]['wav_data'] = frame

    return results


from . import pyaudio
from . import socket
from . import torchaudio
