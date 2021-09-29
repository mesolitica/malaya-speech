import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
from sklearn.utils import shuffle
from glob import glob
import malaya_speech.train as train
import malaya_speech.config
from malaya_speech.train.model import aligner, fastvc, fastspeech
from unidecode import unidecode
from scipy.stats import betabinom
import malaya_speech
import tensorflow as tf
import numpy as np
import re
import json
import malaya

_pad = 'pad'
_start = 'start'
_eos = 'eos'
_punctuation = "!'(),.:;? "
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
maxlen = 1008
minlen = 32
pad_to = 8
data_min = 1e-2
sr = 22050
total_steps = 300000

MALAYA_SPEECH_SYMBOLS = (
    [_pad, _start, _eos] + list(_special) + list(_punctuation) + list(_letters)
)

normalizer = malaya.normalize.normalizer(date=False, time=False, money=False)


def tts_encode(string: str, add_eos: bool = True):
    r = [MALAYA_SPEECH_SYMBOLS.index(c) for c in string if c in MALAYA_SPEECH_SYMBOLS]
    if add_eos:
        r = r + [MALAYA_SPEECH_SYMBOLS.index('eos')]
    return r


def put_spacing_num(string):
    string = re.sub('[A-Za-z]+', lambda ele: ' ' + ele[0] + ' ', string)
    return re.sub(r'[ ]+', ' ', string).strip()


def convert_to_ascii(string):
    return unidecode(string)


def collapse_whitespace(string):
    return re.sub(_whitespace_re, ' ', string)


def cleaning(string, normalize=True, add_eos=False):
    sequence = []
    string = convert_to_ascii(string)
    string = string.replace('&', ' dan ')
    string = re.sub(r'[ ]+', ' ', string).strip()
    if string[-1] in ['-', ',']:
        string = string[:-1]
    if string[-1] != '.':
        string = string + '.'
    if normalize:
        string = normalizer.normalize(string,
                                      check_english=False,
                                      normalize_entity=False,
                                      normalize_text=False,
                                      normalize_url=True,
                                      normalize_email=True,
                                      normalize_year=True)
        string = string['normalize']
    else:
        string = string
    string = put_spacing_num(string)
    string = ''.join([c for c in string if c in MALAYA_SPEECH_SYMBOLS])
    string = re.sub(r'[ ]+', ' ', string).strip()
    string = string.lower()
    return string, tts_encode(string, add_eos=add_eos)


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    x = np.arange(0, phoneme_count)
    mel_text_probs = []
    for i in range(1, mel_count + 1):
        a, b = scaling_factor * i, scaling_factor * (mel_count + 1 - i)
        mel_i_prob = betabinom(phoneme_count, a, b).pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def generate(file):
    with open(file) as fopen:
        audios = json.load(fopen)
    while True:
        audios = shuffle(audios)
        for i in range(len(audios)):
            try:
                audio, _ = malaya_speech.load(audios[i][0])
                mel = malaya_speech.featurization.universal_mel(audio)
                mel_length = len(mel)
                if mel_length > maxlen or mel_length < minlen:
                    continue
                _, text_input = cleaning(audios[i][1])
                text_input = np.array(text_input)
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
                len_mel = [len(mel)]
                len_text_ids = [len(text_input)]
                prior = beta_binomial_prior_distribution(len(text_input), len(mel)).astype(np.float32)
                yield {
                    'mel': mel,
                    'text_ids': text_input,
                    'len_mel': len_mel,
                    'len_text_ids': len_text_ids,
                    'prior': prior,
                }

            except Exception as e:
                print(e)


def get_dataset(files, batch_size=32, shuffle_size=32, thread_count=24):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'mel': tf.float32,
                'text_ids': tf.int32,
                'len_mel': tf.int32,
                'len_text_ids': tf.int32,
                'prior': tf.float32
            },
            output_shapes={
                'mel': tf.TensorShape([None, 80]),
                'text_ids': tf.TensorShape([None]),
                'len_mel': tf.TensorShape([1]),
                'len_text_ids': tf.TensorShape([1]),
                'prior': tf.TensorShape([None, None]),
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
                'prior': tf.TensorShape([None, None]),
            },
            padding_values={
                'mel': tf.constant(0, dtype=tf.float32),
                'text_ids': tf.constant(0, dtype=tf.int32),
                'len_mel': tf.constant(0, dtype=tf.int32),
                'len_text_ids': tf.constant(0, dtype=tf.int32),
                'prior': tf.constant(0, dtype=tf.float32),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    input_ids = features['text_ids']
    lens = features['len_text_ids'][:, 0]
    mel = features['mel']
    mel_lengths = features['len_mel'][:, 0]
    prior = features['prior']

    config = malaya_speech.config.fastspeech_config
    config = fastspeech.Config(vocab_size=1, **config)
    encoder_mel = fastvc.Decoder(config.decoder_self_attention_params, use_position_embedding=True)

    max_length = tf.cast(tf.reduce_max(mel_lengths), tf.int32)
    attention_mask = tf.sequence_mask(
        lengths=mel_lengths, maxlen=max_length, dtype=tf.float32
    )
    attention_mask.set_shape((None, None))
    denser = tf.keras.layers.Dense(
        units=config.decoder_self_attention_params.hidden_size, dtype=tf.float32,
        name='mel_before'
    )
    mel_ = encoder_mel(denser(mel), attention_mask)
    encoder = aligner.AlignmentEncoder(vocab_size=len(MALAYA_SPEECH_SYMBOLS), vocab_embedding=512)
    attention_mask = tf.expand_dims(tf.math.not_equal(input_ids, 0), -1)
    attn_soft, attn_logprob = encoder(mel_, input_ids, mask=attention_mask, attn_prior=prior)
    # attn_hard = encoder.get_hard_attention(attn_soft, lens, mel_lengths)
    forwardsum_loss = aligner.forwardsum_loss(attn_logprob, lens, mel_lengths)
    # bin_loss = aligner.bin_loss(attn_hard, attn_soft)

    global_step = tf.train.get_or_create_global_step()

    loss = tf.cond(tf.less(global_step, int(0.3 * total_steps)),
                   lambda: forwardsum_loss,
                   lambda: forwardsum_loss)

    tf.identity(loss, 'loss')
    tf.identity(forwardsum_loss, 'forwardsum_loss')
    tf.identity(bin_loss, 'bin_loss')

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('forwardsum_loss', forwardsum_loss)
    tf.summary.scalar('bin_loss', bin_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr=1e-4,
            num_train_steps=total_steps,
            num_warmup_steps=int(0.02 * total_steps),
            end_learning_rate=1e-6,
            weight_decay_rate=0.001,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6,
            clip_norm=1.0,
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
        [
            'loss',
            'forwardsum_loss',
            'bin_loss',
        ],
        every_n_iter=1,
    )
]

train_dataset = get_dataset('force-alignment-malay-tts-dataset.json')

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='fastspeech-decoder-aligner-stt',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=10000,
    max_steps=total_steps,
    train_hooks=train_hooks,
)
