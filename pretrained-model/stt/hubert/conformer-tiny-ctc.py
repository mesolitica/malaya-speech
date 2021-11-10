import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import pyroomacoustics as pra
import numpy as np
from pydub import AudioSegment
from sklearn.utils import shuffle
from glob import glob
import random
import json
from malaya_speech.train.model.conformer.model import Model as ConformerModel
from malaya_speech.train.model import hubert, ctc
import malaya_speech.train as train
import malaya_speech.config
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech
import tensorflow as tf
import os
import string


sr = 16000
maxlen = 18
minlen_text = 1
prob_aug = 0.95

unique_vocab = list(string.ascii_lowercase + string.digits) + [' ']


def augment_room(y, scale=1.0):
    corners = np.array(
        [[0, 0], [0, 5 * scale], [3 * scale, 5 * scale], [3 * scale, 0]]
    ).T
    room = pra.Room.from_corners(
        corners,
        fs=sr,
        materials=pra.Material(0.2, 0.15),
        ray_tracing=True,
        air_absorption=True,
    )
    room.extrude(3.5, materials=pra.Material(0.2, 0.15))
    room.set_ray_tracing(
        receiver_radius=0.5, n_rays=1000, energy_thres=1e-5
    )
    room.add_source([1.5 * scale, 4 * scale, 0.5], signal=y)
    R = np.array([[1.5 * scale], [0.5 * scale], [0.5]])
    room.add_microphone(R)
    room.simulate()
    return room.mic_array.signals[0]


def random_amplitude_threshold(sample, low=1, high=2, threshold=0.4):
    y_aug = sample.copy()
    dyn_change = np.random.uniform(low=low, high=high)
    y_aug[np.abs(y_aug) >= threshold] = (
        y_aug[np.abs(y_aug) >= threshold] * dyn_change
    )
    return np.clip(y_aug, -1, 1)


def add_uniform_noise(
    sample, power=0.01, return_noise=False, scale=False
):
    y_noise = sample.copy()
    noise_amp = power * np.random.uniform() * np.amax(y_noise)
    noise = noise_amp * np.random.normal(size=y_noise.shape[0])
    y_noise = y_noise + noise
    if scale:
        y_noise = y_noise / (np.max(np.abs(y_noise)) + 1e-9)
    if return_noise:
        if scale:
            noise = noise / (np.max(np.abs(y_noise)) + 1e-9)
        return y_noise, noise
    else:
        return y_noise


def calc(signal, add_uniform=True):
    choice = random.randint(0, 10)
    print('choice', choice)
    if choice == 0:
        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain=random.randint(25, 50),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 50),
            negate=1,
        )
    if choice == 1:
        x = augmentation.sox_augment_high(
            signal,
            min_bass_gain=random.randint(25, 70),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 50),
            negate=0,
        )
    if choice == 2:
        x = augmentation.sox_augment_low(
            signal,
            min_bass_gain=random.randint(5, 30),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 50),
            negate=random.randint(0, 1),
        )
    if choice == 3:
        x = augmentation.sox_augment_combine(
            signal,
            min_bass_gain_high=random.randint(25, 70),
            min_bass_gain_low=random.randint(5, 30),
            reverberance=random.randint(0, 80),
            hf_damping=10,
            room_scale=random.randint(0, 90),
        )
    if choice == 4:
        x = augmentation.sox_reverb(
            signal,
            reverberance=random.randint(10, 80),
            hf_damping=10,
            room_scale=random.randint(10, 90),
        )
    if choice == 5:
        x = random_amplitude_threshold(
            signal, threshold=random.uniform(0.35, 0.8)
        )
    if choice == 6:
        x = augmentation.lowpass_filter(
            signal, sr=sr, cutoff=random.randint(200, 551)
        )
    if choice == 7:
        x = augmentation.highpass_filter(
            signal, sr=sr, cutoff=random.randint(551, 1653)
        )
    if choice == 8:
        x = augmentation.bandpass_filter(
            signal,
            sr=sr,
            cutoff_low=random.randint(200, 551),
            cutoff_high=random.randint(551, 1653),
        )
    if choice == 9:
        x = augment_room(signal)
    if choice == 10:
        x = signal

    if choice not in [5] and random.gauss(0.5, 0.14) > 0.6:
        x = random_amplitude_threshold(
            x, low=1.0, high=2.0, threshold=random.uniform(0.6, 0.9)
        )

    if random.gauss(0.5, 0.14) > 0.6 and add_uniform:
        x = add_uniform_noise(x, power=random.uniform(0.005, 0.015))

    return x


def mp3_to_wav(file, sr=sr):
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(sr).set_channels(1)
    sample = np.array(audio.get_array_of_samples())
    return malaya_speech.astype.int_to_float(sample), sr


def generate(file):
    with open(file) as fopen:
        dataset = json.load(fopen)
    audios, cleaned_texts = dataset['X'], dataset['Y']
    while True:
        audios, cleaned_texts = shuffle(audios, cleaned_texts)
        for i in range(len(audios)):
            try:
                if audios[i].endswith('.mp3'):
                    # print('found mp3', audios[i])
                    wav_data, _ = mp3_to_wav(audios[i])
                else:
                    wav_data, _ = malaya_speech.load(audios[i], sr=sr)

                if len(cleaned_texts[i]) < minlen_text:
                    # print(f'skipped text too short {audios[i]}')
                    continue

                if (len(wav_data) / sr) > maxlen:
                    continue

                t = [unique_vocab.index(c) for c in cleaned_texts[i]]

                yield {
                    'waveforms': wav_data,
                    'waveforms_length': [len(wav_data)],
                    'targets': t,
                    'targets_length': [len(t)],
                }
            except Exception as e:
                print(e)


def get_dataset(
    file,
    batch_size=12,
    shuffle_size=20,
    thread_count=24,
    maxlen_feature=1800,
):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'waveforms': tf.float32,
                'waveforms_length': tf.int32,
                'targets': tf.int32,
                'targets_length': tf.int32,
            },
            output_shapes={
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            args=(file,),
        )
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            padding_values={
                'waveforms': tf.constant(0, dtype=tf.float32),
                'waveforms_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int32),
                'targets_length': tf.constant(0, dtype=tf.int32),
            },
        )
        return dataset

    return get


class Encoder:
    def __init__(self, config):
        self.config = config
        self.encoder = ConformerModel(**self.config)

    def __call__(self, x, input_mask, training=True):
        return self.encoder(x, training=training)


total_steps = 2000000


def model_fn(features, labels, mode, params):
    config_conformer = malaya_speech.config.conformer_tiny_encoder_config
    config_conformer['subsampling']['type'] = 'none'
    config_conformer['dropout'] = 0.0
    encoder = Encoder(config_conformer)
    cfg = hubert.HuBERTConfig(
        extractor_mode='layer_norm',
        dropout=0.0,
        attention_dropout=0.0,
        encoder_layerdrop=0.0,
        dropout_input=0.0,
        dropout_features=0.0,
        final_dim=128,
    )
    model = hubert.Model(cfg, encoder, ['pad', 'eos', 'unk'] + [str(i) for i in range(100)])
    X = features['waveforms']
    X_len = features['waveforms_length'][:, 0]
    targets = features['targets']
    targets_int32 = tf.cast(targets, tf.int32)
    targets_length = features['targets_length'][:, 0]
    r = model(X, padding_mask=X_len, features_only=True, mask=False)
    logits = tf.layers.dense(r['x'], len(unique_vocab) + 1)
    seq_lens = tf.reduce_sum(
        tf.cast(tf.logical_not(r['padding_mask']), tf.int32), axis=1
    )
    mean_error, sum_error, sum_weight = ctc.loss.ctc_loss(
        logits, seq_lens, targets_int32, targets_length
    )
    loss = mean_error
    accuracy = ctc.metrics.ctc_sequence_accuracy(
        logits, seq_lens, targets_int32, targets_length,
    )

    tf.identity(loss, 'train_loss')
    tf.identity(accuracy, name='train_accuracy')

    tf.summary.scalar('train_accuracy', accuracy)

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    init_checkpoint = 'hubert-conformer-tiny/model.ckpt-1000000'

    assignment_map, initialized_variable_names = train.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr=5e-5,
            num_train_steps=total_steps,
            num_warmup_steps=100000,
            end_learning_rate=0.0,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            clip_norm=1.0,
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': ctc.metrics.ctc_sequence_accuracy_estimator(
                    logits, seq_lens, targets_int32, targets_length
                )
            },
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter=1
    )
]
train_dataset = get_dataset('bahasa-asr-train-combined.json')
dev_dataset = get_dataset('bahasa-asr-test.json')

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='hubert-conformer-tiny-ctc-char',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=20000,
    max_steps=total_steps,
    eval_fn=dev_dataset,
    train_hooks=train_hooks,
)
