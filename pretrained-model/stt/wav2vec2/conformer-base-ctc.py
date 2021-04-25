import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.config
import malaya_speech.train as train
from malaya_speech.train.model import conformer, ctc
import json
import random
from glob import glob
from sklearn.utils import shuffle
from pydub import AudioSegment
import numpy as np
import pyroomacoustics as pra

config = malaya_speech.config.conformer_base_encoder_config
sr = 16000
maxlen = 18
minlen_text = 1

with open('malaya-speech-sst-vocab.json') as fopen:
    unique_vocab = json.load(fopen) + ['{', '}', '[']

featurizer = malaya_speech.tf_featurization.STTFeaturizer(
    normalize_per_feature = True
)
n_mels = featurizer.num_feature_bins


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

                yield {
                    'waveforms': wav_data,
                    'targets': t,
                    'targets_length': [len(t)],
                }
            except Exception as e:
                print(e)


def mel_augmentation(features):

    features = mask_augmentation.warp_time_pil(features)
    features = mask_augmentation.mask_frequency(features, width_freq_mask = 12)
    features = mask_augmentation.mask_time(
        features, width_time_mask = int(features.shape[0] * 0.05)
    )
    return features


def preprocess_inputs(example):
    s = featurizer.vectorize(example['waveforms'])
    s = tf.reshape(s, (-1, n_mels))
    s = tf.compat.v1.numpy_function(mel_augmentation, [s], tf.float32)
    mel_fbanks = tf.reshape(s, (-1, n_mels))
    length = tf.cast(tf.shape(mel_fbanks)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['inputs'] = mel_fbanks
    example['inputs_length'] = length
    example.pop('waveforms', None)
    return example


def get_dataset(
    file,
    batch_size = 12,
    shuffle_size = 20,
    thread_count = 24,
    maxlen_feature = 1800,
):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'waveforms': tf.float32,
                'targets': tf.int32,
                'targets_length': tf.int32,
            },
            output_shapes = {
                'waveforms': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            args = (file,),
        )
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.map(
            preprocess_inputs, num_parallel_calls = thread_count
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'inputs': tf.TensorShape([None, n_mels]),
                'inputs_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            padding_values = {
                'inputs': tf.constant(0, dtype = tf.float32),
                'inputs_length': tf.constant(0, dtype = tf.int32),
                'targets': tf.constant(0, dtype = tf.int32),
                'targets_length': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


total_steps = 500000


def model_fn(features, labels, mode, params):
    conformer_model = conformer.Model(
        kernel_regularizer = None, bias_regularizer = None, **config
    )
    logits = conformer_model(tf.expand_dims(features['inputs'], -1))
    seq_lens = (
        features['inputs_length'][:, 0]
        // conformer_model.conv_subsampling.time_reduction_factor
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

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    init_checkpoint = 'wav2vec2-base-conformer/model.ckpt-500000'

    assignment_map, initialized_variable_names = train.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr = 5e-5,
            num_train_steps = total_steps,
            num_warmup_steps = 50000,
            end_learning_rate = 0.0,
            weight_decay_rate = 0.01,
            beta_1 = 0.9,
            beta_2 = 0.98,
            epsilon = 1e-6,
            clip_norm = 1.0,
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = loss,
            eval_metric_ops = {
                'accuracy': ctc.metrics.ctc_sequence_accuracy_estimator(
                    logits, targets_int32, seq_lens
                )
            },
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter = 1
    )
]
train_dataset = get_dataset('bahasa-asr-train.json')
dev_dataset = get_dataset('bahasa-asr-test.json')

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = 'wav2vec2-conformer-base-ctc',
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 5000,
    max_steps = total_steps,
    eval_fn = dev_dataset,
    train_hooks = train_hooks,
)
