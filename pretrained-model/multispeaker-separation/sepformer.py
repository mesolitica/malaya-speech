import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
import malaya_speech
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from malaya_speech.train.model import fastsplit, fastspeech, sepformer
from malaya_speech.train.model.transformer import encoder
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.utils import tf_featurization
import malaya_speech.train as train
import random
from glob import glob
from sklearn.utils import shuffle

sr = 8000
speakers_size = 4


def get_data(combined_path, speakers_size = 4, sr = 8000):
    combined, _ = malaya_speech.load(combined_path, sr = sr, scale = False)
    y = []
    for i in range(speakers_size):
        y_, _ = malaya_speech.load(
            combined_path.replace('combined', str(i)), sr = sr, scale = False
        )
        y.append(y_)
    return combined, y


def generate():
    combined = glob('split-speaker-8k-train/combined/*.wav')
    while True:
        combined = shuffle(combined)
        for i in range(len(combined)):
            x, y = get_data(combined[i])
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)
            yield {'combined': x, 'y': y, 'length': [len(x)]}


def get_dataset(batch_size = 2):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'combined': tf.float32, 'y': tf.float32, 'length': tf.int32},
            output_shapes = {
                'combined': tf.TensorShape([None, 1]),
                'y': tf.TensorShape([speakers_size, None, 1]),
                'length': tf.TensorShape([None]),
            },
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'combined': tf.TensorShape([None, 1]),
                'y': tf.TensorShape([speakers_size, None, 1]),
                'length': tf.TensorShape([None]),
            },
            padding_values = {
                'combined': tf.constant(0, dtype = tf.float32),
                'y': tf.constant(0, dtype = tf.float32),
                'length': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


total_steps = 10000000


def model_fn(features, labels, mode, params):
    lengths = features['length'][:, 0]
    Y = features['y'][:, :, :, 0]
    config = malaya_speech.config.transformer_config
    dim = 256
    config['hidden_size'] = dim
    config['num_hidden_layers'] = 8
    config['attention_dropout'] = 0.0
    config['relu_dropout'] = 0.0
    config['filter_size'] = 1024
    config['norm_before'] = True
    transformer = lambda: encoder.Encoder(config, train = True)
    model = sepformer.Model(transformer, transformer)
    logits = model(features['combined'])
    estimate_source = tf.transpose(logits[:, :, :, 0], [1, 0, 2])
    loss, max_snr, _ = sepformer.calculate_loss(
        Y, estimate_source, lengths, C = speakers_size
    )
    tf.identity(loss, 'total_loss')
    tf.summary.scalar('total_loss', loss)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value = 1e-4, shape = [], dtype = tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        total_steps,
        end_learning_rate = 0.0,
        power = 1.0,
        cycle = False,
    )
    tf.summary.scalar('learning_rate', learning_rate)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


train_hooks = [tf.train.LoggingTensorHook(['total_loss'], every_n_iter = 1)]
train_dataset = get_dataset()

save_directory = 'split-speaker-sepformer'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 3000,
    max_steps = total_steps,
    train_hooks = train_hooks,
    eval_step = 0,
)
