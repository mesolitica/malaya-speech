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
    eval_fn = None,
    train_hooks = None,
):
    tf.logging.set_verbosity(tf.logging.INFO)

    if num_gpus > 1:
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus = num_gpus,
            auto_shard_dataset = True,
            cross_device_ops = AllReduceCrossDeviceOps(
                'nccl', num_packs = num_gpus
            ),
        )
    else:
        dist_strategy = None

    run_config = RunConfig(
        train_distribute = dist_strategy,
        eval_distribute = dist_strategy,
        log_step_count_steps = log_step,
        model_dir = model_dir,
        save_checkpoints_steps = save_checkpoint_step,
        save_summary_steps = summary_step,
    )
    estimator = tf.estimator.Estimator(
        model_fn = model_fn, params = {}, config = run_config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
    )

    if not eval_fn:
        eval_fn = train_fn

    eval_spec = tf.estimator.EvalSpec(
        input_fn = eval_fn, steps = eval_step, throttle_secs = eval_throttle
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


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
