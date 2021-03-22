import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
from glob import glob
import random
import tensorflow as tf
from malaya_speech.train.model import autovc
from math import ceil
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
import malaya_speech.train as train
import malaya_speech
from collections import defaultdict
import sklearn

speaker_model = malaya_speech.speaker_vector.deep_model('vggvox-v2')

dari_pasentran = glob(
    '/home/husein/speech-bahasa/dari-pasentran-ke-istana/*/*.wav'
)
turki = glob('/home/husein/speech-bahasa/turki/*/*.wav')
salina = glob('/home/husein/speech-bahasa/salina/*/*.wav')

husein = glob('/home/husein/speech-bahasa/audio-wattpad/*.wav')
husein.extend(glob('/home/husein/speech-bahasa/audio-iium/*.wav'))
husein.extend(glob('/home/husein/speech-bahasa/audio/*.wav'))

haqkiem = glob('/home/husein/speech-bahasa/haqkiem/*.wav')
vctk = glob('vtck/**/*.flac', recursive = True)

vctk_speakers = defaultdict(list)
for f in vctk:
    s = f.split('/')[-1].split('_')[0]
    vctk_speakers[s].append(f)

speakers = []

for s in vctk_speakers.keys():
    speakers.extend(
        random.sample(vctk_speakers[s], min(500, len(vctk_speakers[s])))
    )

files = dari_pasentran + turki + salina + husein + haqkiem + speakers
sr = 22050


def pad_seq(x, base = 32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), x.shape[0]


def generate(batch_max_steps = 24576, hop_size = 256):
    while True:
        shuffled = sklearn.utils.shuffle(files)
        for f in shuffled:
            audio, _ = malaya_speech.load(f, sr = sr)
            mel = malaya_speech.featurization.universal_mel(audio)

            batch_max_frames = batch_max_steps // hop_size

            if len(mel) > batch_max_frames:
                interval_start = 0
                interval_end = len(mel) - batch_max_frames
                start_frame = random.randint(interval_start, interval_end)
                start_step = start_frame * hop_size
                mel = mel[start_frame : start_frame + batch_max_frames, :]

            mel, len_pad = pad_seq(mel)

            v = speaker_model([audio])

            yield {
                'mel': mel,
                'mel_length': [len(mel)],
                'audio': audio,
                'v': v[0],
            }


def get_dataset(batch_size = 4):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'mel_length': tf.int32,
                'audio': tf.float32,
                'v': tf.float32,
            },
            output_shapes = {
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'audio': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
        )
        dataset = dataset.shuffle(batch_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'audio': tf.TensorShape([None]),
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
            padding_values = {
                'audio': tf.constant(0, dtype = tf.float32),
                'mel': tf.constant(0, dtype = tf.float32),
                'mel_length': tf.constant(0, dtype = tf.int32),
                'v': tf.constant(0, dtype = tf.float32),
            },
        )
        return dataset

    return get


total_steps = 300000


def model_fn(features, labels, mode, params):
    vectors = features['v'] * 30 - 3.5
    mels = features['mel']
    mels_len = features['mel_length'][:, 0]
    dim_neck = 32
    bottleneck = 512
    model = autovc.Model(
        dim_neck = dim_neck, dim_pre = bottleneck, freq = dim_neck
    )
    encoder_outputs, mel_before, mel_after, codes = model(
        mels, vectors, vectors
    )
    codes_ = model.call_second(mel_after, vectors)
    loss_f = tf.losses.mean_squared_error
    max_length = tf.cast(tf.reduce_max(mels_len), tf.int32)
    mask = tf.sequence_mask(
        lengths = mels_len, maxlen = max_length, dtype = tf.float32
    )
    mask = tf.expand_dims(mask, axis = -1)
    mel_loss_before = loss_f(
        labels = mels, predictions = mel_before, weights = mask
    )
    mel_loss_after = loss_f(
        labels = mels, predictions = mel_after, weights = mask
    )
    g_loss_cd = tf.losses.absolute_difference(codes, codes_)
    loss = mel_loss_before + mel_loss_after + g_loss_cd

    tf.identity(loss, 'total_loss')
    tf.identity(mel_loss_before, 'mel_loss_before')
    tf.identity(mel_loss_after, 'mel_loss_after')
    tf.identity(g_loss_cd, 'g_loss_cd')

    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('mel_loss_before', mel_loss_before)
    tf.summary.scalar('mel_loss_after', mel_loss_after)
    tf.summary.scalar('g_loss_cd', g_loss_cd)

    global_step = tf.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['total_loss', 'mel_loss_before', 'mel_loss_after', 'g_loss_cd'],
        every_n_iter = 1,
    )
]
train_dataset = get_dataset()

save_directory = 'autovc-vggvox-v2'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 2000,
    max_steps = total_steps,
    train_hooks = train_hooks,
    eval_step = 0,
)
