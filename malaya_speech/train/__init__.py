import tensorflow as tf
from tensorflow.python.distribute.cross_device_ops import (
    AllReduceCrossDeviceOps,
)
from tensorflow.python.estimator.run_config import RunConfig
from . import augmentation
from . import accuracy
from . import audio_encoder
from . import char_encoder
from . import loss


def run_training(
    train_fn,
    model_fn,
    model_dir,
    num_gpus = 1,
    log_step = 100,
    summary_step = 100,
    save_checkpoint_step = 1000,
    max_steps = 10000,
    eval_step = 10,
    eval_throttle = 120,
    eval_fn = None,
    hooks = None,
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
        input_fn = train_fn, max_steps = max_steps, hooks = hooks
    )

    if not eval_fn:
        eval_fn = train_fn

    eval_spec = tf.estimator.EvalSpec(
        input_fn = eval_fn, steps = eval_step, throttle_secs = eval_throttle
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
