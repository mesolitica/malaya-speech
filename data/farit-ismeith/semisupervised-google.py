from glob import glob
import malaya_speech
from malaya_speech import Pipeline
import numpy as np
from malaya_speech.utils import generator
from pydub import AudioSegment
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.system('mkdir output-wav output-text')

import speech_recognition as sr
import soundfile as sf
import pyrubberband as pyrb

r = sr.Recognizer()

vad = malaya_speech.vad.webrtc(minimum_amplitude=200)
model = malaya_speech.noise_reduction.deep_model(model='resnet-unet')

mp3s = glob('*.mp3')
len(mp3s)

from malaya_speech.model.frame import Frame
from malaya_speech.utils.group import (
    combine_frames,
    group_frames,
    group_frames_threshold,
)
from malaya_speech.utils.combine import without_silent


def split_vad_duration(
    frames,
    max_duration: float = 5.0,
    negative_threshold: float = 0.1,
):
    """
    Split a sample into multiple samples based maximum duration of voice activities.

    Parameters
    ----------
    frames: List[Tuple[Frame, label]]
    max_duration: float, optional (default = 5.0)
        Maximum duration to assume one sample combined from voice activities.
    negative_threshold: float, optional (default = 0.1)
        If `negative_threshold` is 0.1, means that, length negative samples must at least 0.1 second.

    Returns
    -------
    result : List[Frame]
    """
    grouped = group_frames(frames)
    grouped = group_frames_threshold(
        grouped, threshold_to_stop=negative_threshold
    )
    results, temp, lengths = [], [], 0
    for no, g in enumerate(grouped):
        a = g[0]
        l = a.duration
        lengths += l
        temp.append(a)
        if lengths >= max_duration:
            results.append(combine_frames(temp))
            temp = []
            lengths = 0

    if len(temp):
        results.append(combine_frames(temp))
    return results


p = Pipeline()

pipeline_left = (
    p.map(generator.frames, frame_duration_ms=30, sample_rate=44100)
)

pipeline_right = (
    p.map(malaya_speech.resample, old_samplerate=44100, new_samplerate=16000)
    .map(malaya_speech.astype.float_to_int)
    .map(generator.frames, frame_duration_ms=30, sample_rate=16000,
         append_ending_trail=False)
    .foreach_map(vad)
)

pipeline_left.foreach_zip(pipeline_right).map(without_silent, silent_trail=2000)

p_noise = Pipeline()
p_noise_pipeline = (
    p_noise.map(generator.frames, frame_duration_ms=15000, sample_rate=44100)
    .foreach_map(model)
    .foreach_map(lambda x: x['voice'])
    .map(np.concatenate)
)


def split(file, max_duration=10.0):
    print(file)
    audio = AudioSegment.from_mp3(file).set_channels(1)
    y = np.array(audio.get_array_of_samples())
    y = malaya_speech.astype.int_to_float(y)
    y = p_noise(y)['concatenate']
    y_int = malaya_speech.astype.float_to_int(y)
    y_ = malaya_speech.resample(y_int, audio.frame_rate, 16000).astype(int)
    frames = generator.frames(y, 30, audio.frame_rate)
    frames_ = generator.frames(
        y_, 30, 16000, append_ending_trail=False
    )
    frames_webrtc = [
        (frames[no], vad(frame)) for no, frame in enumerate(frames_)
    ]
    splitted = split_vad_duration(
        frames_webrtc,
        max_duration=max_duration,
        negative_threshold=0.1,
    )
    results = [s.array for s in splitted]
    return results, audio, audio.frame_rate


def audiosegment_google_speech(audio, filename, sample_rate, lang='ms'):
    if os.path.exists('output-wav/' + filename):
        return False

    sf.write(filename, audio, sample_rate)
    try:
        with sr.AudioFile(filename) as source:
            a = r.record(source)

        text = r.recognize_google(a, language=lang)
    except BaseException:
        text = ''

    if len(text):
        text_filename = f'output-text/{filename}.txt'
        with open(text_filename, 'w') as fopen:
            fopen.write(text)

        sf.write('output-wav/' + filename, audio, 44100)

    os.remove(filename)

    return True


for no, filename in enumerate(mp3s):
    wavs = glob('output-wav/*.wav')
    if any([filename in w for w in wavs]):
        print(f'skip {filename}, exist')
        continue
    try:
        audios, audio, sample_rate = split(filename)
        for part in tqdm(range(len(audios))):
            temp_filename = f'{filename}-part-{part}.wav'
            audiosegment_google_speech(audios[part], temp_filename, 44100)
    except Exception as e:
        print(e)

    print(f'DONE: {no + 1} / {len(mp3s)}')
