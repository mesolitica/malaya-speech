import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')

import tensorflow as tf
import malaya_speech
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.train.model import fastsplit, fastspeech, fastvc
from malaya_speech.train.model import sepformer_old as sepformer
from malaya_speech.utils import tf_featurization
import malaya_speech.train as train
import random
import pickle
from glob import glob
from sklearn.utils import shuffle

sr = 22050
speakers_size = 4


def get_data(combined_path, speakers_size=4, sr=22050):
    with open(combined_path, 'rb') as fopen:
        combined = pickle.load(fopen)
    y = []
    for i in range(speakers_size):
        with open(combined_path.replace('combined', str(i)), 'rb') as fopen:
            y_ = pickle.load(fopen)
        y.append(y_)
    return combined, y


def to_mel(y):
    mel = malaya_speech.featurization.universal_mel(y)
    mel[mel <= np.log(1e-2)] = np.log(1e-2)
    return mel


def generate():
    combined = glob('split-speaker-22k-train/combined/*.pkl')
    while True:
        combined = shuffle(combined)
        for i in range(len(combined)):
            x, y = get_data(combined[i])
            yield {'combined': x, 'y': y, 'length': [len(x)]}


def get_dataset(batch_size=8):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'combined': tf.float32, 'y': tf.float32, 'length': tf.int32},
            output_shapes={
                'combined': tf.TensorShape([None, 80]),
                'y': tf.TensorShape([speakers_size, None, 80]),
                'length': tf.TensorShape([None]),
            },
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'combined': tf.TensorShape([None, 80]),
                'y': tf.TensorShape([speakers_size, None, 80]),
                'length': tf.TensorShape([None]),
            },
            padding_values={
                'combined': tf.constant(np.log(1e-2), dtype=tf.float32),
                'y': tf.constant(np.log(1e-2), dtype=tf.float32),
                'length': tf.constant(0, dtype=tf.int32),
            },
        )
        return dataset

    return get


total_steps = 10000000


def model_fn(features, labels, mode, params):
    lengths = features['length'][:, 0]
    config = malaya_speech.config.fastspeech_config
    dim = 256
    config['encoder_hidden_size'] = dim
    config['decoder_hidden_size'] = dim
    config['encoder_num_hidden_layers'] = 4
    config['encoder_num_attention_heads'] = 4
    config = fastspeech.Config(vocab_size=1, **config)

    def transformer(): return sepformer.Encoder_FastSpeech(
        config.encoder_self_attention_params
    )
    def decoder(): return fastvc.Decoder(config.decoder_self_attention_params)
    model = sepformer.Model_Mel(
        transformer, transformer, decoder, activation=None
    )
    logits = model(features['combined'], lengths)
    outputs = tf.transpose(logits, [1, 2, 0, 3])
    loss = fastsplit.calculate_loss(
        features['y'], outputs, lengths, C=speakers_size
    )
    tf.identity(loss, 'total_loss')
    tf.summary.scalar('total_loss', loss)

    global_step = tf.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:

        train_op = train.optimizer.adamw.create_optimizer(
            loss,
            init_lr=0.0001,
            num_train_steps=total_steps,
            num_warmup_steps=100000,
            end_learning_rate=0.00001,
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


train_hooks = [tf.train.LoggingTensorHook(['total_loss'], every_n_iter=1)]
train_dataset = get_dataset()

save_directory = 'split-speaker-sepformer-mel'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=3000,
    max_steps=total_steps,
    train_hooks=train_hooks,
    eval_step=0,
)
