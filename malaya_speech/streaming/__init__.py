from malaya_speech.utils.validator import check_pipeline
from scipy.io.wavfile import write
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


def stream(
    audio_class,
    vad_model=None,
    asr_model=None,
    classification_model=None,
    sample_rate: int = 16000,
    segment_length: int = 2560,
    num_padding_frames: int = 20,
    ratio: float = 0.75,
    min_length: float = 0.1,
    max_length: float = 10.0,
    filename: str = None,
    realtime_print: bool = True,
    **kwargs,
):

    if vad_model:
        check_pipeline(vad_model, 'vad', 'vad')
    if asr_model:
        check_pipeline(asr_model, 'speech-to-text', 'asr_model')
    if classification_model:
        check_pipeline(
            classification_model, 'classification', 'classification_model'
        )

    audio = audio_class(
        vad_model=vad_model,
        sample_rate=sample_rate,
        segment_length=segment_length,
        **kwargs,
    )
    frames = audio.vad_collector(num_padding_frames=num_padding_frames, ratio=ratio)

    results = []
    wav_data = np.array([], dtype=np.float32)
    length = 0
    count = 0

    try:
        for frame in frames:

            if frame is not None:
                length += frame.shape[0] / sample_rate
                wav_data = np.concatenate([wav_data, frame])

            if frame is None and length >= min_length or length >= max_length:
                now = datetime.now()
                data_dict = {
                    'wav_data': wav_data,
                    'timestamp': now,
                }
                t = ''

                if asr_model:
                    t_ = asr_model(wav_data)
                    if isinstance(t_, dict):
                        t_ = t_['speech-to-text']

                    data_dict['asr_model'] = t_
                    logger.info(f'Sample asr_model {count} {now}: {t_}')

                    t += t_ + ' '

                if classification_model:
                    t_ = classification_model(wav_data)
                    if isinstance(t_, dict):
                        t_ = t_['classification']

                    data_dict['classification_model'] = t_
                    logger.info(f'Sample classification_model {count} {now}: {t_}')

                    t += f'({t_})' + ' '

                if realtime_print and len(t):
                    print(t, end='', flush=True)

                results.append(data_dict)
                wav_data = np.array([], dtype=np.float32)
                length = 0
                count += 1

    except KeyboardInterrupt:

        if filename is not None:
            logger.info(f'saved audio to {filename}')
            write(filename, np.concatenate([r[0] for r in results]))

    except Exception as e:
        raise e

    audio.destroy()
    return results


from . import pyaudio
from . import torchaudio
