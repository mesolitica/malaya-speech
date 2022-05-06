import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import malaya_speech
import malaya_speech.train as train
from malaya_speech.train.model.conformer.model import Model as ConformerModel
from malaya_speech.train.model import hubert
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
maxlen = 15
minlen = 2
kmean = hubert.kmeans.ApplyKmeans_TF('kmean.km')


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
                'waveforms_length': [len(wav_data)],
                'Y': [y],
            }


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
            {'waveforms': tf.float32,
             'waveforms_length': tf.int32,
             'Y': tf.int32,
             },
            output_shapes={
                'waveforms': tf.TensorShape([None]),
                'waveforms_length': tf.TensorShape([None]),
                'Y': tf.TensorShape([None]),
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
                'Y': tf.TensorShape([None]),
            },
            padding_values={
                'waveforms': tf.constant(0, dtype=tf.float32),
                'waveforms_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int32),
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

    seq = r['x']
    Y = features['Y']
    first_token_tensor = tf.squeeze(seq[:, 0:1, :], axis=1)
    pooled_output = tf.keras.layers.Dense(embedding_dim, activation='tanh',
                                          use_bias=True, trainable=True)(first_token_tensor)
    logits = tf.keras.layers.Dense(num_class, trainable=True,)(pooled_output)
    entropy_speakers = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=Y
        )
    )

    loss = entropy_m * 0.95 + entropy_u * 0.05 + entropy_speakers

    tf.identity(entropy_m, 'entropy_m')
    tf.summary.scalar('entropy_m', entropy_m)

    tf.identity(entropy_u, 'entropy_u')
    tf.summary.scalar('entropy_u', entropy_u)

    tf.identity(loss, 'train_loss')

    accuracy = tf.metrics.accuracy(
        labels=Y, predictions=tf.argmax(logits, axis=1)
    )

    tf.identity(accuracy[1], name='train_accuracy')

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
        ['entropy_m', 'entropy_u', 'entropy_speakers', 'train_accuracy', 'train_loss'], every_n_iter=1
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
