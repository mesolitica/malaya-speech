import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from glob import glob
import random
import tensorflow as tf
from malaya_speech.train.model import fastspeech, conformervc
from math import ceil
from functools import partial
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss
import malaya_speech.train as train
import malaya_speech
import random
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
vctk = glob('vtck/**/*.flac', recursive=True)

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

config_conformer = malaya_speech.config.conformer_base_encoder_config
config_conformer['dmodel'] = 384
config_conformer['num_blocks'] = 8
config_conformer['head_size'] = 18
config_conformer['kernel_size'] = 3
config_conformer['subsampling']['type'] = 'none'

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 10e-9},
    'lr_policy_params': {
        'warmup_steps': 60000,
        'max_lr': (0.05 / config_conformer['dmodel']),
    },
}


def transformer_schedule(step, d_model, warmup_steps=6000, max_lr=None):
    arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
    arg2 = step * (warmup_steps ** -1.5)
    arg1 = tf.cast(arg1, tf.float32)
    arg2 = tf.cast(arg2, tf.float32)
    lr = tf.math.rsqrt(tf.cast(d_model, tf.float32)) * tf.math.minimum(
        arg1, arg2
    )
    if max_lr is not None:
        max_lr = tf.cast(max_lr, tf.float32)
        return tf.math.minimum(max_lr, lr)
    return lr


def learning_rate_scheduler(global_step):

    return transformer_schedule(
        tf.cast(global_step, tf.float32),
        config_conformer['dmodel'],
        **parameters['lr_policy_params'],
    )


def generate(hop_size=256):
    while True:
        shuffled = sklearn.utils.shuffle(files)
        for f in shuffled:
            audio, _ = malaya_speech.load(f, sr=sr)
            mel = malaya_speech.featurization.universal_mel(audio)

            batch_max_steps = random.randint(16384, 110_250)
            batch_max_frames = batch_max_steps // hop_size

            if len(mel) > batch_max_frames:
                interval_start = 0
                interval_end = len(mel) - batch_max_frames
                start_frame = random.randint(interval_start, interval_end)
                start_step = start_frame * hop_size
                audio = audio[start_step: start_step + batch_max_steps]
                mel = mel[start_frame: start_frame + batch_max_frames, :]

            v = speaker_model([audio])

            yield {
                'mel': mel,
                'mel_length': [len(mel)],
                'audio': audio,
                'v': v[0],
            }


def get_dataset(batch_size=8):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'mel_length': tf.int32,
                'audio': tf.float32,
                'v': tf.float32,
            },
            output_shapes={
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'audio': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
        )
        dataset = dataset.shuffle(batch_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'audio': tf.TensorShape([None]),
                'mel': tf.TensorShape([None, 80]),
                'mel_length': tf.TensorShape([None]),
                'v': tf.TensorShape([512]),
            },
            padding_values={
                'audio': tf.constant(0, dtype=tf.float32),
                'mel': tf.constant(0, dtype=tf.float32),
                'mel_length': tf.constant(0, dtype=tf.int32),
                'v': tf.constant(0, dtype=tf.float32),
            },
        )
        return dataset

    return get


total_steps = 300_000


def model_fn(features, labels, mode, params):
    vectors = features['v'] * 30 - 3.5
    mels = features['mel']
    mels_len = features['mel_length'][:, 0]
    dim_neck = 32
    config = malaya_speech.config.fastspeech_config
    config['encoder_hidden_size'] = 512 + 80
    config['decoder_hidden_size'] = 512 + dim_neck
    config = fastspeech.Config(vocab_size=1, **config)
    model = conformervc.model.Model(dim_neck, config_conformer, config, 80)
    encoder_outputs, mel_before, mel_after, codes = model(
        mels, vectors, vectors, mels_len
    )
    codes_ = model.call_second(mel_after, vectors, mels_len)
    loss_f = tf.losses.absolute_difference
    max_length = tf.cast(tf.reduce_max(mels_len), tf.int32)
    mask = tf.sequence_mask(
        lengths=mels_len, maxlen=max_length, dtype=tf.float32
    )
    mask = tf.expand_dims(mask, axis=-1)
    mel_loss_before = loss_f(
        labels=mels, predictions=mel_before, weights=mask
    )
    mel_loss_after = loss_f(
        labels=mels, predictions=mel_after, weights=mask
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

        lr = learning_rate_scheduler(global_step)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            loss, global_step=global_step
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['total_loss', 'mel_loss_before', 'mel_loss_after', 'g_loss_cd'],
        every_n_iter=1,
    )
]
train_dataset = get_dataset()

save_directory = 'conformervc-base-32-vggvox-v2'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=2000,
    max_steps=total_steps,
    train_hooks=train_hooks,
    eval_step=0,
)
