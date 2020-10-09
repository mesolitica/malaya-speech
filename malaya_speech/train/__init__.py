import tensorflow as tf
from tensorflow.python.distribute.cross_device_ops import (
    AllReduceCrossDeviceOps,
)
from tensorflow.python.estimator.run_config import RunConfig
from herpetologist import check_type
from typing import List, Dict


from . import accuracy
from . import loss
from . import prepare_data
from . import utils


@check_type
def run_training(
    train_fn,
    model_fn,
    model_dir: str,
    num_gpus: int = 1,
    log_step: int = 100,
    summary_step: int = 100,
    save_checkpoint_step: int = 1000,
    max_steps: int = 10000,
    eval_step: int = 10,
    eval_throttle: int = 120,
    use_tpu: bool = False,
    tpu_name: str = None,
    tpu_zone: str = None,
    gcp_project: str = None,
    iterations_per_loop: int = 100,
    num_tpu_cores: int = 8,
    train_batch_size: int = 128,
    train_hooks = None,
    eval_fn = None,
):
    tf.logging.set_verbosity(tf.logging.INFO)

    if num_gpus > 1 and not use_tpu:
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus = num_gpus,
            auto_shard_dataset = True,
            cross_device_ops = AllReduceCrossDeviceOps(
                'nccl', num_packs = num_gpus
            ),
        )
    else:
        dist_strategy = None

    if use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone = tpu_zone, project = gcp_project
        )
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster = tpu_cluster_resolver,
            master = None,
            model_dir = model_dir,
            save_checkpoints_steps = save_checkpoint_step,
            tpu_config = tf.contrib.tpu.TPUConfig(
                iterations_per_loop = iterations_per_loop,
                num_shards = num_tpu_cores,
                per_host_input_for_training = is_per_host,
            ),
        )
    else:

        run_config = RunConfig(
            train_distribute = dist_strategy,
            eval_distribute = dist_strategy,
            log_step_count_steps = log_step,
            model_dir = model_dir,
            save_checkpoints_steps = save_checkpoint_step,
            save_summary_steps = summary_step,
        )

    if use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu = use_tpu,
            model_fn = model_fn,
            config = run_config,
            train_batch_size = train_batch_size,
            eval_batch_size = None,
        )
        eval_fn = None

    else:

        estimator = tf.estimator.Estimator(
            model_fn = model_fn, params = {}, config = run_config
        )

    if eval_fn:
        train_spec = tf.estimator.TrainSpec(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn = eval_fn, steps = eval_step, throttle_secs = eval_throttle
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    else:
        estimator.train(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )


@check_type
def prepare_dataset(
    generator,
    data_dir: str,
    shards: List[Dict],
    prefix: str = 'dataset',
    shuffle: bool = True,
    already_shuffled: bool = False,
):
    prepare_data.check_shard(shards)
    filepath_fns = {
        'train': prepare_data.training_filepaths,
        'dev': prepare_data.dev_filepaths,
        'test': prepare_data.test_filepaths,
    }

    split_paths = [
        (
            split['split'],
            filepath_fns[split['split']](
                prefix, data_dir, split['shards'], shuffled = already_shuffled
            ),
        )
        for split in shards
    ]
    all_paths = []
    for _, paths in split_paths:
        all_paths.extend(paths)

    prepare_data.generate_files(generator, all_paths)

    if shuffle:
        prepare_data.shuffle_dataset(all_paths)
