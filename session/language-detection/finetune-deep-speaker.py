import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../gcs/mesolitica-storage.json'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import malaya_speech.train as train
import malaya_speech.train.model.deep_speaker as deep_speaker
import malaya_speech

DIMENSION = 64


def calc(v):

    r = malaya_speech.utils.featurization.read_mfcc(v, voice_only = False)
    return r


def preprocess_inputs(example):
    s = tf.compat.v1.numpy_function(calc, [example['inputs']], tf.float32)

    s = tf.reshape(s, (-1, DIMENSION, 1))
    example['inputs'] = s

    return example


def parse(serialized_example):

    data_fields = {
        'inputs': tf.VarLenFeature(tf.float32),
        'targets': tf.VarLenFeature(tf.int64),
    }
    features = tf.parse_single_example(
        serialized_example, features = data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    features = preprocess_inputs(features)

    keys = list(features.keys())
    for k in keys:
        if k not in ['inputs', 'targets']:
            features.pop(k, None)

    return features


def get_dataset(files, batch_size = 32, shuffle_size = 1024, thread_count = 24):
    def get():
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(parse, num_parallel_calls = thread_count)
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'inputs': tf.TensorShape([None, DIMENSION, 1]),
                'targets': tf.TensorShape([None]),
            },
            padding_values = {
                'inputs': tf.constant(0, dtype = tf.float32),
                'targets': tf.constant(0, dtype = tf.int64),
            },
        )
        dataset = dataset.repeat()
        return dataset

    return get


def model_fn(features, labels, mode, params):
    learning_rate = (1e-5,)
    init_checkpoint = '../deep-speaker/out/vggvox.ckpt'
    Y = tf.cast(features['targets'][:, 0], tf.int32)

    model = deep_speaker.model.Model(
        features['inputs'], num_class = 7, mode = 'train'
    )
    logits = model.logits

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = Y
        )
    )

    tf.identity(loss, 'train_loss')

    accuracy = tf.metrics.accuracy(
        labels = Y, predictions = tf.argmax(logits, axis = 1)
    )

    tf.identity(accuracy[1], name = 'train_accuracy')

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables = [v for v in variables if 'prediction' not in v.name]

    assignment_map, initialized_variable_names = train.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = loss,
            eval_metric_ops = {'accuracy': accuracy},
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter = 1
    )
]

files = tf.io.gfile.glob(
    'gs://mesolitica-general/language-detection/data/*.tfrecords'
)
train_dataset = get_dataset(files)

save_directory = 'output-deep-speaker-language-detection'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 25000,
    max_steps = 300000,
    train_hooks = train_hooks,
)
