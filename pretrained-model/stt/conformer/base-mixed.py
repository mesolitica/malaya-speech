import pyroomacoustics as pra
import numpy as np
from pydub import AudioSegment
from sklearn.utils import shuffle
from glob import glob
import random
import json
import malaya_speech.train as train
import malaya_speech.config
import malaya_speech.train.model.transducer as transducer
import malaya_speech.train.model.conformer as conformer
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


subwords = malaya_speech.subword.load('transducer-mixed.subword')
config = malaya_speech.config.conformer_base_encoder_config
sr = 16000
maxlen = 18
minlen_text = 1
prob_aug = 0.85

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 10e-9},
    'lr_policy_params': {
        'warmup_steps': 40000,
        'max_lr': (0.05 / config['dmodel']),
    },
}

featurizer = malaya_speech.tf_featurization.STTFeaturizer(
    normalize_per_feature=True
)
n_mels = featurizer.num_feature_bins


def transformer_schedule(step, d_model, warmup_steps=4000, max_lr=None):
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


def mel_augmentation(features):

    features = mask_augmentation.warp_time_pil(features)
    features = mask_augmentation.mask_frequency(features, width_freq_mask=12)
    features = mask_augmentation.mask_time(
        features, width_time_mask=int(features.shape[0] * 0.05)
    )
    return features


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

                if (len(wav_data) / sr) > maxlen:
                    # print(f'skipped audio too long {audios[i]}')
                    continue

                if len(cleaned_texts[i]) < minlen_text:
                    # print(f'skipped text too short {audios[i]}')
                    continue

                t = malaya_speech.subword.encode(
                    subwords, cleaned_texts[i], add_blank=False
                )

                if random.random() > prob_aug:
                    wav_data = calc(wav_data)

                back = np.zeros(shape=(2000,))
                front = np.zeros(shape=(200,))
                wav_data = np.concatenate([front, wav_data, back], axis=-1)

                yield {
                    'waveforms': wav_data,
                    'targets': t,
                    'targets_length': [len(t)],
                }
            except Exception as e:
                print(e)


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
    batch_size=16,
    shuffle_size=16,
    thread_count=24,
    maxlen_feature=1800,
):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'waveforms': tf.float32,
                'targets': tf.int32,
                'targets_length': tf.int32,
            },
            output_shapes={
                'waveforms': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            args=(file,),
        )
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.map(
            preprocess_inputs, num_parallel_calls=thread_count
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'inputs': tf.TensorShape([None, n_mels]),
                'inputs_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            padding_values={
                'inputs': tf.constant(0, dtype=tf.float32),
                'inputs_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int32),
                'targets_length': tf.constant(0, dtype=tf.int32),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    conformer_model = conformer.Model(
        kernel_regularizer=None, bias_regularizer=None, **config
    )
    decoder_config = malaya_speech.config.conformer_base_decoder_config
    transducer_model = transducer.rnn.Model(
        conformer_model, vocabulary_size=subwords.vocab_size, **decoder_config
    )
    targets_length = features['targets_length'][:, 0]
    v = tf.expand_dims(features['inputs'], -1)
    z = tf.zeros((tf.shape(features['targets'])[0], 1), dtype=tf.int32)
    c = tf.concat([z, features['targets']], axis=1)

    logits = transducer_model([v, c, targets_length + 1], training=True)

    cost = transducer.loss.rnnt_loss(
        logits=logits,
        labels=features['targets'],
        label_length=targets_length,
        logit_length=features['inputs_length'][:, 0]
        // conformer_model.conv_subsampling.time_reduction_factor,
    )
    mean_error = tf.reduce_mean(cost)

    loss = mean_error

    tf.identity(loss, 'train_loss')

    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # variables = [v for v in variables if 'transducer_prediction' in v.name]
    # init_checkpoint = 'transducer-rnn-base/model-rename.ckpt'

    # assignment_map, initialized_variable_names = train.get_assignment_map_from_checkpoint(
    #     variables, init_checkpoint
    # )

    # tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.optimize_loss(
            loss,
            tf.train.AdamOptimizer,
            parameters['optimizer_params'],
            learning_rate_scheduler,
            summaries=['learning_rate', 'loss_scale'],
            larc_params=parameters.get('larc_params', None),
            loss_scaling=parameters.get('loss_scaling', 1.0),
            loss_scaling_params=parameters.get('loss_scaling_params', None),
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [tf.train.LoggingTensorHook(['train_loss'], every_n_iter=1)]
train_dataset = get_dataset('mixed-asr-train.json')
dev_dataset = get_dataset('mixed-asr-test.json')

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='asr-base-mixed-conformer-transducer-v2',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=25000,
    max_steps=1000_000,
    eval_fn=dev_dataset,
    train_hooks=train_hooks,
)
