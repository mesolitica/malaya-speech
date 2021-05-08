import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.augmentation.spectrogram as mask_augmentation
from malaya_speech.train.model import conformer, ctc, leaf
import malaya_speech.config
import malaya_speech.train as train
import json
import numpy as np
import random
from glob import glob
from sklearn.utils import shuffle
from pydub import AudioSegment
import numpy as np
import pyroomacoustics as pra

sr = 16000
maxlen = 16
minlen_text = 1
config = malaya_speech.config.conformer_base_encoder_config

with open('malaya-speech-sst-vocab.json') as fopen:
    unique_vocab = json.load(fopen) + ['{', '}', '[']

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 10e-9},
    'lr_policy_params': {
        'warmup_steps': 40000,
        'max_lr': (0.05 / config['dmodel']),
    },
}

featurizer = malaya_speech.tf_featurization.STTFeaturizer(
    normalize_per_feature = True
)
n_mels = featurizer.num_feature_bins


def transformer_schedule(step, d_model, warmup_steps = 4000, max_lr = None):
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
        config['dmodel'],
        **parameters['lr_policy_params'],
    )


def augment_room(y, scale = 1.0):
    corners = np.array(
        [[0, 0], [0, 5 * scale], [3 * scale, 5 * scale], [3 * scale, 0]]
    ).T
    room = pra.Room.from_corners(
        corners,
        fs = sr,
        materials = pra.Material(0.2, 0.15),
        ray_tracing = True,
        air_absorption = True,
    )
    room.extrude(3.5, materials = pra.Material(0.2, 0.15))
    room.set_ray_tracing(
        receiver_radius = 0.5, n_rays = 1000, energy_thres = 1e-5
    )
    room.add_source([1.5 * scale, 4 * scale, 0.5], signal = y)
    R = np.array([[1.5 * scale], [0.5 * scale], [0.5]])
    room.add_microphone(R)
    room.simulate()
    return room.mic_array.signals[0]


def mel_augmentation(features):

    features = mask_augmentation.warp_time_pil(features)
    features = mask_augmentation.mask_frequency(features, width_freq_mask = 12)
    features = mask_augmentation.mask_time(
        features, width_time_mask = int(features.shape[0] * 0.05)
    )
    return features


def mp3_to_wav(file, sr = sr):
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
                    wav_data, _ = malaya_speech.load(audios[i], sr = sr)

                if len(cleaned_texts[i]) < minlen_text:
                    # print(f'skipped text too short {audios[i]}')
                    continue

                if (len(wav_data) / sr) > maxlen:
                    continue

                if random.random() > 0.9:
                    wav_data = augment_room(wav_data)

                t = [unique_vocab.index(c) for c in cleaned_texts[i]]

                # while True:

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
    batch_size = 8,
    shuffle_size = 20,
    thread_count = 24,
    maxlen_feature = 1800,
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
            output_shapes = {
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            args = (file,),
        )
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            padding_values = {
                'waveforms': tf.constant(0, dtype = tf.float32),
                'waveforms_length': tf.constant(0, dtype = tf.int32),
                'targets': tf.constant(0, dtype = tf.int32),
                'targets_length': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    X = features['waveforms']
    X_len = features['waveforms_length'][:, 0]
    training = True
    batch_size = tf.shape(X)[0]
    leaf_featurizer = leaf.Model(
        n_filters = 40, preemp = False, mean_var_norm = False
    )
    leaf_f = leaf_featurizer(X, training = training)
    # pretty hacky
    padded_lens = tf.cast(
        tf.cast(X_len, tf.float32) // 159.734_042_553_191_5, tf.int32
    )
    conformer_model = conformer.Model(
        kernel_regularizer = None, bias_regularizer = None, **config
    )
    targets_length = features['targets_length'][:, 0]
    v = tf.expand_dims(leaf_f, -1)

    logits = tf.layers.dense(
        conformer_model(v, training = training), len(unique_vocab) + 1
    )
    seq_lens = (
        padded_lens // conformer_model.conv_subsampling.time_reduction_factor
    )

    targets_int32 = tf.cast(features['targets'], tf.int32)
    mean_error, sum_error, sum_weight = ctc.loss.ctc_loss(
        logits, targets_int32, seq_lens
    )
    loss = mean_error
    accuracy = ctc.metrics.ctc_sequence_accuracy(
        logits, targets_int32, seq_lens
    )

    tf.identity(loss, 'train_loss')
    tf.identity(accuracy, name = 'train_accuracy')

    tf.summary.scalar('train_accuracy', accuracy)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.optimize_loss(
            loss,
            tf.train.AdamOptimizer,
            parameters['optimizer_params'],
            learning_rate_scheduler,
            summaries = ['learning_rate', 'loss_scale'],
            larc_params = parameters.get('larc_params', None),
            loss_scaling = parameters.get('loss_scaling', 1.0),
            loss_scaling_params = parameters.get('loss_scaling_params', None),
        )
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
        ['train_accuracy', 'train_loss'], every_n_iter = 1
    )
]
train_dataset = get_dataset('bahasa-asr-train.json')

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = 'asr-leaf-base-conformer-ctc',
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 5000,
    max_steps = 500_000,
    train_hooks = train_hooks,
)
