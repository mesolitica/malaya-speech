import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/husein/t5/prepare/mesolitica-tpu.json'

from pydub import AudioSegment
import pyroomacoustics as pra
import numpy as np
import json
import malaya_speech.train as train
import malaya_speech.config
import malaya_speech.train.model.transducer as transducer
import malaya_speech.train.model.conformer as conformer
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech
import tensorflow as tf
import random
import string

char_vocabs = [''] + list(string.ascii_lowercase + string.digits) + [' ']
subwords_malay = malaya_speech.subword.load('bahasa-512.subword')
subwords_singlish = malaya_speech.subword.load('singlish-512.subword')
subwords_mandarin = malaya_speech.subword.load('mandarin-512.subword')
langs = [subwords_malay, subwords_singlish, subwords_mandarin]
len_vocab = [l.vocab_size for l in langs]
config = malaya_speech.config.conformer_base_encoder_config
sr = 16000
maxlen_subwords = 100
minlen_text = 1
maxlen = 16
prob_aug = 0.9

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
    if random.random() < prob_aug:
        return signal
    try:
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

        return x.astype(np.float32)
    except:
        return signal


def mel_augmentation(features):

    features = mask_augmentation.warp_time_pil(features)
    features = mask_augmentation.mask_frequency(features, width_freq_mask=12)
    features = mask_augmentation.mask_time(
        features, width_time_mask=int(features.shape[0] * 0.05)
    )
    return features


def get_subwords(ids, lang):
    lang = lang[0]
    text = ''.join([char_vocabs[c] for c in ids])
    t = malaya_speech.subword.encode(
        langs[lang], text, add_blank=False
    )
    t = np.array(t) + sum(len_vocab[:lang])
    return t.astype(np.int32)


def preprocess_inputs(example):
    s = tf.compat.v1.numpy_function(calc, [example['waveforms']], tf.float32)
    s = tf.reshape(s, (-1,))
    s = featurizer.vectorize(s)
    s = tf.reshape(s, (-1, n_mels))
    s = tf.compat.v1.numpy_function(mel_augmentation, [s], tf.float32)
    mel_fbanks = tf.reshape(s, (-1, n_mels))
    length = tf.cast(tf.shape(mel_fbanks)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['inputs'] = mel_fbanks
    example['inputs_length'] = length

    t = tf.compat.v1.numpy_function(get_subwords, [example['targets'], example['lang']], tf.int32)
    t = tf.reshape(t, (-1,))
    example['targets'] = t
    length = tf.cast(tf.shape(t)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['targets_length'] = length

    return example


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.compat.v1.VarLenFeature(tf.float32),
        'targets': tf.compat.v1.VarLenFeature(tf.int64),
        'targets_length': tf.compat.v1.VarLenFeature(tf.int64),
        'lang': tf.compat.v1.VarLenFeature(tf.int64),
    }
    features = tf.compat.v1.parse_single_example(
        serialized_example, features=data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    features = preprocess_inputs(features)

    keys = list(features.keys())
    for k in keys:
        if k not in ['waveforms', 'inputs', 'inputs_length', 'targets', 'targets_length']:
            features.pop(k, None)

    return features


def pop(features):
    features.pop('waveforms', None)
    return features


def get_dataset(files, batch_size=10, shuffle_size=32, num_cpu_threads=6,
                thread_count=24, is_training=True):
    def get():
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(files))
            cycle_length = min(num_cpu_threads, len(files))
            d = d.interleave(
                tf.data.TFRecordDataset,
                cycle_length=cycle_length,
                block_length=batch_size)
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(files)
            d = d.repeat()
        d = d.map(parse, num_parallel_calls=thread_count)
        d = d.filter(
            lambda x: tf.less(tf.shape(x['waveforms'])[0] / sr, maxlen)
        )
        d = d.filter(
            lambda x: tf.less(tf.shape(x['targets'])[0], maxlen_subwords)
        )
        d = d.map(pop, num_parallel_calls=thread_count)
        d = d.padded_batch(
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
        return d

    return get


def model_fn(features, labels, mode, params):
    conformer_model = conformer.Model(
        kernel_regularizer=None, bias_regularizer=None, **config
    )
    decoder_config = malaya_speech.config.conformer_base_decoder_config
    transducer_model = transducer.rnn.Model(
        conformer_model,
        vocabulary_size=sum(len_vocab), **decoder_config
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

with open('3mixed-train-test.json') as fopen:
    dataset = json.load(fopen)

train_dataset = get_dataset(dataset['train'], is_training=True)
dev_dataset = get_dataset(dataset['test'], is_training=False)

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='asr-base-conformer-transducer-3mixed',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=25000,
    max_steps=2000_000,
    eval_fn=dev_dataset,
    train_hooks=train_hooks,
)
