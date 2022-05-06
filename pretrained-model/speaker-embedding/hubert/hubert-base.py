import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import malaya_speech
import malaya_speech.train as train
from malaya_speech.train.model.conformer.model import Model as ConformerModel
from malaya_speech.train.model import hubert
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import string
import json
import random
from glob import glob

unique_vocab = [''] + list(string.ascii_lowercase + string.digits) + [' ']

with open('/home/husein/youtube/voxceleb2-label.json') as fopen:
    ids = json.load(fopen)

num_class = len(ids)

train_set = glob('/home/husein/youtube/voxceleb-dev/*.wav')
test_set = glob('/home/husein/youtube/voxceleb-wav/*.wav')

sr = 16000
maxlen = 18
minlen = 3
weight_decay = 1e-5


def generate(files):
    while True:
        random.shuffle(files)
        for f in files:
            f = f.decode() if isinstance(f, bytes) else f
            x, _ = malaya_speech.load(f)
            label = os.path.split(f)[1].replace('wav-', '').split('-')[1]
            y = int(ids[label])

            len_x = len(x)

            if (len_x / sr) < minlen:
                continue

            if (len_x / sr) > maxlen:
                x = augmentation.random_sampling(x, sr, random.randint(1000 * minlen, 1000 * maxlen))

            yield {
                'waveforms': x,
                'waveforms_length': [len(x)],
                'Y': [y],
            }


def get_dataset(files, batch_size=4, shuffle_size=32, thread_count=24):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'waveforms': tf.float32,
                'waveforms_length': tf.int32,
                'Y': tf.int32,
            },
            output_shapes={
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'Y': tf.TensorShape([None]),
            },
            args=(files,),
        )
        dataset = dataset.filter(
            lambda x: tf.less(tf.shape(x['waveforms'])[0] / sr, maxlen)
        )
        dataset = dataset.filter(
            lambda x: tf.greater(tf.shape(x['waveforms'])[0] / sr, minlen)
        )
        dataset = dataset.padded_batch(
            shuffle_size,
            padded_shapes={
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'Y': tf.TensorShape([None]),
            },
            padding_values={
                'waveforms': tf.constant(0, dtype=tf.float32),
                'waveforms_length': tf.constant(0, dtype=tf.int32),
                'Y': tf.constant(0, dtype=tf.int32),
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


total_steps = 3000000


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def model_fn(features, labels, mode, params):
    config_conformer = malaya_speech.config.conformer_base_encoder_config
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
        final_dim=256,
    )
    model = hubert.Model(cfg, encoder, ['pad', 'eos', 'unk'] + [str(i) for i in range(100)])
    X = features['waveforms']
    X_len = features['waveforms_length'][:, 0]
    Y = features['Y']
    Y_onehot = tf.one_hot(Y, depth=num_class)

    r = model(X, padding_mask=X_len, features_only=True, mask=False)
    first_token_tensor = tf.squeeze(r['x'][:, 0:1, :], axis=1)
    pooled_output = keras.layers.Dense(cfg.final_dim * 2, activation='tanh',
                                       kernel_initializer='orthogonal',
                                       use_bias=True, trainable=True,
                                       kernel_regularizer=keras.regularizers.l2(weight_decay),
                                       bias_regularizer=keras.regularizers.l2(weight_decay))(first_token_tensor)
    logits = keras.layers.Dense(num_class,
                                kernel_initializer='orthogonal',
                                use_bias=False, trainable=True,
                                kernel_constraint=keras.constraints.unit_norm(),
                                kernel_regularizer=keras.regularizers.l2(weight_decay),
                                bias_regularizer=keras.regularizers.l2(weight_decay),
                                name='prediction')(pooled_output)
    loss = tf.reduce_mean(amsoftmax_loss(Y_onehot, logits))
    accuracy = tf.metrics.accuracy(
        labels=Y, predictions=tf.argmax(logits, axis=1)
    )

    tf.identity(accuracy[1], name='train_accuracy')

    tf.identity(loss, 'train_loss')

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    init_checkpoint = 'hubert-conformer-base-output-3mixed/model.ckpt-2000000'

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
            eval_metric_ops={'accuracy': accuracy},
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter=1
    )
]


train_dataset = get_dataset(train_set)
test_dataset = get_dataset(test_set)

save_directory = 'hubert-base-voxceleb2'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=25000,
    max_steps=total_steps,
    train_hooks=train_hooks,
)
