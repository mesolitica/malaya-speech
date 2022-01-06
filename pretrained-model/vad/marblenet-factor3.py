import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import malaya_speech.train as train
import malaya_speech
import malaya_speech.train.model.marblenet as marblenet
import tensorflow as tf
import malaya_speech.config

config = malaya_speech.config.ctc_featurizer_config
config['feature_type'] = 'mfcc'
# config['num_feature_bins'] = 64
# config['normalize_per_feature'] = True
featurizer = malaya_speech.tf_featurization.STTFeaturizer(**config)
n_mels = featurizer.num_feature_bins

parameters = {
    'optimizer_params': {},
    'lr_policy_params': {
        'learning_rate': 1e-3,
        'min_lr': 1e-5,
        'warmup_steps': 0,
        'decay_steps': 2000_000,
    },
}


def learning_rate_scheduler(global_step):
    return train.schedule.cosine_decay(
        global_step, **parameters['lr_policy_params']
    )


def preprocess_inputs(example):
    s = featurizer.vectorize(example['waveforms'])
    s = tf.reshape(s, (-1, n_mels))
    length = tf.cast(tf.shape(s)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['inputs'] = s
    example['inputs_length'] = length

    return example


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.VarLenFeature(tf.float32),
        'targets': tf.VarLenFeature(tf.int64),
    }
    features = tf.parse_single_example(
        serialized_example, features=data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    features = preprocess_inputs(features)

    keys = list(features.keys())
    for k in keys:
        if k not in ['inputs', 'inputs_length', 'targets']:
            features.pop(k, None)

    return features


def get_dataset(files, batch_size=32, shuffle_size=32, num_cpu_threads=6,
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
        d = d.padded_batch(
            batch_size,
            padded_shapes={
                'inputs': tf.TensorShape([None, n_mels]),
                'inputs_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
            },
            padding_values={
                'inputs': tf.constant(0, dtype=tf.float32),
                'inputs_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int64),
            },
        )
        return d

    return get


def model_fn(features, labels, mode, params):

    print(features)
    Y = tf.cast(features['targets'][:, 0], tf.int32)
    model = marblenet.Model(features['inputs'], features['inputs_length'][:, 0], factor=3, training=True)
    logits = tf.reduce_sum(model.logits['outputs'], axis=1)
    logits = tf.layers.dense(logits, 2)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=Y
        )
    )
    tf.identity(loss, 'train_loss')

    accuracy = tf.metrics.accuracy(
        labels=Y, predictions=tf.argmax(logits, axis=1)
    )

    tf.identity(accuracy[1], name='train_accuracy')

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

train_files = tf.io.gfile.glob(
    'vad/data/vad-train*'
)
train_dataset = get_dataset(train_files, is_training=True)

dev_files = tf.io.gfile.glob(
    'vad/data/vad-dev*'
)
dev_dataset = get_dataset(dev_files, is_training=False)

save_directory = 'marblenet-factor3'

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir=save_directory,
    num_gpus=1,
    log_step=1,
    max_steps=parameters['lr_policy_params']['decay_steps'],
    save_checkpoint_step=25000,
    eval_fn=None,
    train_hooks=train_hooks,
)
