import tensorflow as tf
import re
import collections
import six
import optimization
from malaya_speech.train.model import wav2vec2, bert, fastspeech

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file',
    None,
    'Input TF example files (can be a glob or comma separated).',
)

flags.DEFINE_string(
    'output_dir',
    None,
    'The output directory where the model checkpoints will be written.',
)

flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')

flags.DEFINE_float('learning_rate', 5e-4, 'The initial learning rate.')

flags.DEFINE_integer('num_train_steps', 1000000, 'Number of training steps.')

flags.DEFINE_integer('num_warmup_steps', 10000, 'Number of warmup steps.')

flags.DEFINE_integer(
    'save_checkpoints_steps', 10000, 'How often to save the model checkpoint.'
)

flags.DEFINE_integer(
    'iterations_per_loop', 100, 'How many steps to make in each estimator call.'
)

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or GPU/CPU.')

tf.flags.DEFINE_string(
    'tpu_name',
    None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.',
)

tf.flags.DEFINE_string(
    'tpu_zone',
    None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.',
)

tf.flags.DEFINE_string(
    'gcp_project',
    None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.',
)

tf.flags.DEFINE_string('master', None, '[Optional] TensorFlow master URL.')

flags.DEFINE_integer(
    'num_tpu_cores',
    8,
    'Only used if `use_tpu` is True. Total number of TPU cores to use.',
)

flags.DEFINE_bool('do_train', True, 'Whether to run training.')

flags.DEFINE_string('init_checkpoint', None, 'Initial checkpoint.')


def input_fn_builder(input_files, is_training, num_cpu_threads = 4):

    data_fields = {
        'waveforms': tf.VarLenFeature(tf.float32),
        'waveforms_len': tf.VarLenFeature(tf.int64),
    }

    def parse(serialized_example):

        features = tf.parse_single_example(
            serialized_example, features = data_fields
        )
        for k in features.keys():
            features[k] = features[k].values

        features['waveforms'].set_shape((10 * 16000,))
        features['waveforms_len'].set_shape((1,))
        return features

    def input_fn(params):
        batch_size = params['batch_size']

        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size = len(input_files))
            cycle_length = min(num_cpu_threads, len(input_files))
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy = is_training,
                    cycle_length = cycle_length,
                )
            )
            d = d.shuffle(buffer_size = 100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
        d = d.map(parse, num_parallel_calls = 32)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, data_fields),
                batch_size = batch_size,
                num_parallel_batches = num_cpu_threads,
                drop_remainder = True,
            )
        )
        return d

    return input_fn


def _decode_record(example, name_to_features):
    """Decodes a record to a TensorFlow example."""

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


class Encoder:
    def __init__(self, config):
        self.config = config
        self.model = None

    def __call__(self, x, input_mask, training = True):
        if self.model is None:
            input_mask = tf.logical_not(input_mask)
            self.model = bert.BertModel(
                config = self.config,
                is_training = training,
                input_ids = x,
                input_mask = input_mask,
            )
        return self.model.sequence_output


def model_fn_builder(
    init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, use_tpu
):
    def model_fn(features, labels, mode, params):
        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info(
                '  name = %s, shape = %s' % (name, features[name].shape)
            )

        X = features['waveforms']
        X_len = features['waveforms_len'][:, 0]
        encoder = Encoder(config = bert.BertConfig())
        cfg = wav2vec2.Wav2Vec2Config()
        model = wav2vec2.Model(cfg, encoder)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        r, num_vars, curr_temp = model(X, padding_mask = X_len)

        logits = r['x']
        logits = tf.transpose(logits, [2, 1, 0])
        logits = tf.reshape(logits, (-1, tf.shape(logits)[-1]))
        target = tf.zeros(
            shape = (tf.shape(r['x'])[1] * tf.shape(r['x'])[2]),
            dtype = tf.int32,
        )
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = target, logits = logits
        )
        loss = tf.reduce_sum(loss)

        extra_losses = []
        if 'prob_perplexity' in r:
            extra_losses.append((num_vars - r['prob_perplexity']) / num_vars)

        if 'features_pen' in r:
            extra_losses.append(r['features_pen'])

        sample_size = tf.cast(tf.shape(target)[0], tf.float32)

        loss_weights = [0.1, 10]
        for p, coef in zip(extra_losses, loss_weights):
            if coef != 0 and p is not None:
                p = coef * p * sample_size
                loss += p
                
        total_loss = loss
        tf.identity(total_loss, 'train_loss')

        scaffold_fn = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss,
                learning_rate,
                num_train_steps,
                num_warmup_steps,
                use_tpu,
            )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,
                loss = total_loss,
                train_op = train_op,
                scaffold_fn = scaffold_fn,
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode, loss = total_loss, scaffold_fn = scaffold_fn
            )
        else:
            raise ValueError(
                'Only TRAIN and EVAL modes are supported: %s' % (mode)
            )

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.'
        )

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info('*** Input Files ***')
    for input_file in input_files:
        tf.logging.info('  %s' % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone = FLAGS.tpu_zone, project = FLAGS.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster = tpu_cluster_resolver,
        master = FLAGS.master,
        model_dir = FLAGS.output_dir,
        save_checkpoints_steps = FLAGS.save_checkpoints_steps,
        tpu_config = tf.contrib.tpu.TPUConfig(
            iterations_per_loop = FLAGS.iterations_per_loop,
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host,
        ),
    )

    model_fn = model_fn_builder(
        init_checkpoint = FLAGS.init_checkpoint,
        learning_rate = FLAGS.learning_rate,
        num_train_steps = FLAGS.num_train_steps,
        num_warmup_steps = FLAGS.num_warmup_steps,
        use_tpu = FLAGS.use_tpu,
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = FLAGS.train_batch_size,
        eval_batch_size = FLAGS.train_batch_size,
    )

    if FLAGS.do_train:
        tf.logging.info('***** Running training *****')
        tf.logging.info('  Batch size = %d', FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files = input_files, is_training = True
        )
        estimator.train(
            input_fn = train_input_fn, max_steps = FLAGS.num_train_steps
        )


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_dir')
    tf.app.run()
