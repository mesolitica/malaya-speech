import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
import numpy as np
from glob import glob
from itertools import cycle
import json

with open('mels-male.json') as fopen:
    files = json.load(fopen)

reduction_factor = 1
maxlen = 1008
minlen = 32
pad_to = 2
data_min = 1e-2

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_rejected = '\'():;"'

MALAYA_SPEECH_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 1e-9},
    'lr_policy_params': {
        'warmup_steps': 4000,
        'learning_rate': 1.0,
    },
}

config = glowtts.Config(mel=80, vocabs=len(MALAYA_SPEECH_SYMBOLS))


def noam_schedule(step, channels, learning_rate=1.0, warmup_steps=4000):
    return learning_rate * channels ** -0.5 * \
        tf.minimum(step ** -0.5, step * warmup_steps ** -1.5)


def learning_rate_scheduler(global_step):
    return noam_schedule(
        tf.cast(global_step, tf.float32),
        config.channels,
        **parameters['lr_policy_params'],
    )


total_steps = 100_000


def generate(files):
    file_cycle = cycle(files)
    while True:
        f = next(file_cycle).decode()
        mel = np.load(f)
        mel_length = len(mel)
        f_wav = f.replace('/mels', '/audios')
        wav = np.load(f_wav)
        if mel_length > maxlen or mel_length < minlen:
            continue

        stop_token_target = np.zeros([len(mel)], dtype=np.float32)

        text_ids = np.load(f.replace('mels', 'text_ids'), allow_pickle=True)[
            0
        ]
        text_ids = ''.join(
            [
                c
                for c in text_ids
                if c in MALAYA_SPEECH_SYMBOLS and c not in _rejected
            ]
        )
        text_ids = re.sub(r'[ ]+', ' ', text_ids).strip()
        text_input = np.array(
            [MALAYA_SPEECH_SYMBOLS.index(c) for c in text_ids]
        )
        num_pad = pad_to - ((len(text_input) + 2) % pad_to)
        text_input = np.pad(
            text_input, ((1, 1)), 'constant', constant_values=((1, 2))
        )
        text_input = np.pad(
            text_input, ((0, num_pad)), 'constant', constant_values=0
        )
        num_pad = pad_to - ((len(mel) + 1) % pad_to) + 1
        pad_value_mel = np.log(data_min)
        mel = np.pad(
            mel,
            ((0, num_pad), (0, 0)),
            'constant',
            constant_values=pad_value_mel,
        )
        num_pad = pad_to - len(wav) % pad_to
        wav = np.pad(
            wav,
            ((0, num_pad)),
            'constant',
        )
        len_mel = [len(mel)]
        len_text_ids = [len(text_input)]

        yield {
            'mel': mel,
            'text_ids': text_input,
            'len_mel': len_mel,
            'len_text_ids': len_text_ids,
            'f': [f],
            'audio': wav,
        }


def get_dataset(files, batch_size=32, shuffle_size=32, thread_count=24):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'text_ids': tf.int32,
                'len_mel': tf.int32,
                'len_text_ids': tf.int32,
                'f': tf.string,
                'audio': tf.float32,
            },
            output_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'f': tf.TensorShape([1]),
                'audio': tf.TensorShape([None]),
            },
            args=(files,),
        )
        dataset = dataset.padded_batch(
            shuffle_size,
            padded_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'f': tf.TensorShape([1]),
                'audio': tf.TensorShape([None]),
            },
            padding_values={
                'mel': tf.constant(0, dtype=tf.float32),
                'text_ids': tf.constant(0, dtype=tf.int32),
                'len_mel': tf.constant(0, dtype=tf.int32),
                'len_text_ids': tf.constant(0, dtype=tf.int32),
                'f': tf.constant('', dtype=tf.string),
                'audio': tf.constant(0, dtype=tf.float32),
            },
        )
        return dataset

    return get


features = get_dataset(files['train'])().make_one_shot_iterator().get_next()
features

features = dataset.make_one_shot_iterator().get_next()
features

import malaya_speech
import malaya_speech.train
from malaya_speech.train.model import vits, melgan, hifigan
from malaya_speech.train.model.vits.slicing import rand_slice_segments
from malaya_speech.train.model import stft
import malaya_speech.config
from malaya_speech.train.loss import calculate_2d_loss, calculate_3d_loss

segment_size = 8192
hop_size = 256
config = vits.Config(mel=80, vocabs=len(MALAYA_SPEECH_SYMBOLS))

hifigan_config = malaya_speech.config.hifigan_config
generator = hifigan.Generator(
    hifigan.GeneratorConfig(**hifigan_config['hifigan_generator_params']),
    name='hifigan_generator',
)
multiperiod_discriminator = hifigan.MultiPeriodDiscriminator(
    hifigan.DiscriminatorConfig(
        **hifigan_config['hifigan_discriminator_params']
    ),
    name='hifigan_multiperiod_discriminator',
)
multiscale_discriminator = melgan.MultiScaleDiscriminator(
    melgan.DiscriminatorConfig(
        **hifigan_config['melgan_discriminator_params'],
        name='melgan_multiscale_discriminator',
    )
)
discriminator = hifigan.Discriminator(
    multiperiod_discriminator, multiscale_discriminator
)

stft_loss = stft.loss.MultiResolutionSTFT(**hifigan_config['stft_loss_params'])
mels_loss = melgan.loss.TFMelSpectrogram()
mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()

_, losses, attn = model.compute_loss(text=text, textlen=text_lengths, mel=mel_outputs, mellen=mel_lengths)
