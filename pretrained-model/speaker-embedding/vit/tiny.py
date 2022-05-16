import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from sklearn.utils import shuffle
from glob import glob
import random
import json
import malaya_speech.train as train
import malaya_speech.config
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.train.model import vit
import malaya_speech
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

config = malaya_speech.config.vit_tiny_config
config['dim'] = 384
config['mlp_ratio'] = 4
config['image_height'] = 80
config['image_width'] = 620

sr = 16000
maxlen = 18
minlen = 2
embedding_dim = 512
data_min = np.log(1e-2)

with open('/home/husein/youtube/voxceleb2-label.json') as fopen:
    ids = json.load(fopen)

num_class = len(ids)

train_set = glob('/home/husein/youtube/voxceleb-dev/*.wav')

featurizer = malaya_speech.tf_featurization.STTFeaturizer(
    normalize_per_feature=True
)
n_mels = featurizer.num_feature_bins


def mel_augmentation(features):

    features = mask_augmentation.warp_time_pil(features)
    features = mask_augmentation.mask_frequency(features, width_freq_mask=12)
    features = mask_augmentation.mask_time(
        features, width_time_mask=int(features.shape[0] * 0.05)
    )
    return features


def generate(files):
    while True:
        random.shuffle(files)
        for f in files:
            f = f.decode() if isinstance(f, bytes) else f
            wav_data, _ = malaya_speech.load(f)
            label = os.path.split(f)[1].replace('wav-', '').split('-')[1]
            y = int(ids[label])

            len_x = len(wav_data) / sr

            if len_x < minlen:
                continue

            if len_x > maxlen:
                wav_data = augmentation.random_sampling(wav_data, sr, random.randint(1000 * minlen, 1000 * maxlen))

            yield {
                'waveforms': wav_data,
                'targets': [y],
            }


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
    batch_size=64,
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
            },
            output_shapes={
                'waveforms': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
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
            },
            padding_values={
                'inputs': tf.constant(np.log(data_min), dtype=tf.float32),
                'inputs_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int32),
            },
        )
        return dataset

    return get


total_steps = 2000_000


def model_fn(features, labels, mode, params):
    model = vit.Model(**config)
    X = tf.expand_dims(features['inputs'], -1)
    X_resize = tf.image.resize(X, (config['image_width'], config['image_height']),)
    seq = model(X_resize, training=True)
    Y = features['targets'][:, 0]
    first_token_tensor = tf.squeeze(seq[:, 0:1, :], axis=1)
    pooled_output = keras.layers.Dense(embedding_dim, activation='tanh',
                                       use_bias=True, trainable=True)(first_token_tensor)
    logits = keras.layers.Dense(num_class, trainable=True,)(pooled_output)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=Y
        )
    )
    accuracy = tf.metrics.accuracy(
        labels=Y, predictions=tf.argmax(logits, axis=1)
    )

    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(loss, 'train_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr=2e-5,
            num_train_steps=total_steps,
            num_warmup_steps=20000,
            end_learning_rate=0.0,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            clip_norm=3.0,
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'accuracy': accuracy},
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter=1
    )
]


train_dataset = get_dataset(train_set)

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='vit-tiny-voxceleb',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=25000,
    max_steps=total_steps,
    train_hooks=train_hooks,
)
