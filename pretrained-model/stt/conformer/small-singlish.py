import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/husein/t5/prepare/mesolitica-tpu.json'

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
import wandb
wandb.init()

subwords = malaya_speech.subword.load('transducer-singlish.subword')
config = malaya_speech.config.conformer_small_encoder_config
sr = 16000
maxlen = 18
maxlen_subwords = 100
minlen_text = 1

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


def mel_augmentation(features):

    features = mask_augmentation.warp_time_pil(features)
    features = mask_augmentation.mask_frequency(features, width_freq_mask=12)
    features = mask_augmentation.mask_time(
        features, width_time_mask=int(features.shape[0] * 0.05)
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
    example['targets'] = tf.cast(example['targets'], tf.int32)
    example['targets_length'] = tf.cast(example['targets_length'], tf.int32)
    return example


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.compat.v1.VarLenFeature(tf.float32),
        'targets': tf.compat.v1.VarLenFeature(tf.int64),
        'targets_length': tf.compat.v1.VarLenFeature(tf.int64),
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


def get_dataset(files, batch_size=20, shuffle_size=32, num_cpu_threads=4,
                thread_count=24, is_training=True):
    def get():
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(files))
            cycle_length = min(num_cpu_threads, len(files))
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
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
    decoder_config = malaya_speech.config.conformer_small_decoder_config
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


train_hooks = [tf.train.LoggingTensorHook(['train_loss'], every_n_iter=1),
               wandb.tensorflow.WandbHook(steps_per_log=1000)]

with open('imda-tfrecords.json') as fopen:
    imda_tfrecord = json.load(fopen)

train_dataset = get_dataset(imda_tfrecord['train'], is_training=True)
dev_dataset = get_dataset(imda_tfrecord['test'], is_training=False)

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='asr-small-conformer-transducer-singlish',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=25000,
    max_steps=1000_000,
    eval_fn=dev_dataset,
    train_hooks=train_hooks,
)
