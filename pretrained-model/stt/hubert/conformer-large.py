import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import malaya_speech
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech.config
import malaya_speech.train as train
from malaya_speech.train.model import hubert
from malaya_speech.train.model.conformer.model import Model as ConformerModel
import json
import random
from glob import glob
from sklearn.utils import shuffle
from pydub import AudioSegment
import numpy as np
import pyroomacoustics as pra

sr = 16000
maxlen = 18
minlen_text = 1
kmean = hubert.kmeans.ApplyKmeans_TF('kmean.km')

config = malaya_speech.config.transducer_featurizer_config
config['feature_type'] = 'mfcc'
config['num_feature_bins'] = 30
config['stride_ms'] = 20
featurizer = malaya_speech.utils.tf_featurization.STTFeaturizer(**config)


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
                    continue
                else:
                    wav_data, _ = malaya_speech.load(audios[i], sr=sr)

                if len(cleaned_texts[i]) < minlen_text:
                    # print(f'skipped text too short {audios[i]}')
                    continue

                if (len(wav_data) / sr) > maxlen:
                    wav_data = wav_data[: sr * maxlen]

                yield {
                    'waveforms': wav_data,
                    'waveforms_length': [len(wav_data)],
                }
            except Exception as e:
                print(e)


def preprocess_inputs(example):
    v = featurizer.vectorize(example['waveforms'])
    deltas = malaya_speech.utils.tf_featurization.deltas(v)
    ddeltas = malaya_speech.utils.tf_featurization.deltas(deltas)
    concated = tf.concat([v, deltas, ddeltas], axis=1)
    s = tf.compat.v1.numpy_function(kmean, [concated], tf.int64)
    s = tf.cast(s, tf.int32)
    kmean_tf = tf.reshape(s, (-1,)) + 3
    example['targets'] = kmean_tf
    return example


def get_dataset(
    file,
    batch_size=4,
    shuffle_size=20,
    thread_count=24,
    maxlen_feature=1800,
):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'waveforms': tf.float32, 'waveforms_length': tf.int32},
            output_shapes={
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
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
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
            },
            padding_values={
                'waveforms': tf.constant(0, dtype=tf.float32),
                'waveforms_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int32),
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


total_steps = 1000000


def model_fn(features, labels, mode, params):
    config_conformer = malaya_speech.config.conformer_large_encoder_config
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
        final_dim=768,
    )
    model = hubert.Model(cfg, encoder, ['pad', 'eos', 'unk'] + [str(i) for i in range(100)])
    X = features['waveforms']
    X_len = features['waveforms_length'][:, 0]
    Y = features['targets']
    r = model(X, padding_mask=X_len, target_list=Y)

    target_m = tf.zeros((tf.shape(r['logit_m_list'])[0],), dtype=tf.int32)
    target_u = tf.zeros((tf.shape(r['logit_u_list'])[0],), dtype=tf.int32)

    sample_size = tf.cast(tf.shape(target_m)[0], tf.float32)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_m, logits=r['logit_m_list'])
    entropy_m = tf.reduce_sum(entropy) / sample_size

    sample_size = tf.cast(tf.shape(target_u)[0], tf.float32)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_u, logits=r['logit_u_list'])
    entropy_u = tf.reduce_sum(entropy) / sample_size

    loss = entropy_m * 0.95 + entropy_u * 0.05

    tf.identity(entropy_m, 'entropy_m')
    tf.summary.scalar('entropy_m', entropy_m)

    tf.identity(entropy_u, 'entropy_u')
    tf.summary.scalar('entropy_u', entropy_u)

    tf.identity(loss, 'train_loss')

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
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['entropy_m', 'entropy_u', 'train_loss'],
        every_n_iter=1,
    )
]
train_dataset = get_dataset('bahasa-asr-train.json')
dev_dataset = get_dataset('bahasa-asr-test.json')

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='hubert-conformer-large',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=5000,
    max_steps=total_steps,
    eval_fn=dev_dataset,
    train_hooks=train_hooks,
)
