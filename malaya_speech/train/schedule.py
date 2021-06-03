import tensorflow as tf


def transformer_schedule(step, d_model, warmup_steps=4000, max_lr=None):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (warmup_steps ** -1.5)
    lr = tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)
    if max_lr is not None:
        return tf.math.minimum(max_lr, lr)
    return lr


def cosine_decay(
    global_step,
    learning_rate,
    decay_steps,
    power=1.0,
    begin_decay_at=0,
    min_lr=0.0,
    warmup_steps=0,
):
    if warmup_steps > 0:
        learning_rate = tf.cond(
            global_step < warmup_steps,
            lambda: (
                learning_rate
                * tf.cast(global_step, tf.float32)
                / tf.cast(warmup_steps, tf.float32)
            ),
            lambda: learning_rate,
        )
    lr = tf.cond(
        global_step < begin_decay_at,
        lambda: learning_rate,
        lambda: tf.train.cosine_decay(
            learning_rate=learning_rate,
            global_step=global_step - begin_decay_at,
            decay_steps=decay_steps,
            alpha=min_lr,
        ),
        name='learning_rate',
    )
    return lr


def inv_poly_decay(
    global_step,
    learning_rate,
    decay_steps,
    min_lr,
    power=1.0,
    begin_decay_at=0,
    warmup_steps=0,
    name='learning_rate',
):
    min_lr = max(min_lr, 1e-8)
    min_lr = min(min_lr, learning_rate)
    if power <= 0.0:
        raise ValueError('Inv poly decay requires power >  0.')
    if global_step is None:
        raise ValueError('Inv poly decay requires global_step')

    with ops.name_scope(name, 'InvDecay', [learning_rate, global_step]) as name:
        scale = (
            math.pow(learning_rate / min_lr, 1.0 / power) - 1.0
        ) / decay_steps

        learning_rate = ops.convert_to_tensor(
            learning_rate, name='learning_rate'
        )

        decay_steps = tf.cast(decay_steps, tf.float32)
        global_step = tf.cast(global_step, tf.float32)
        denom = tf.pow(1.0 + scale * global_step, power)
        lr = tf.div(learning_rate, denom, name=name)

    return lr


def poly_decay(
    global_step,
    learning_rate,
    decay_steps,
    power=1.0,
    begin_decay_at=0,
    min_lr=0.0,
    warmup_steps=0,
):

    if warmup_steps > 0:
        learning_rate = tf.cond(
            global_step < warmup_steps,
            lambda: (
                learning_rate
                * tf.cast(global_step, tf.float32)
                / tf.cast(warmup_steps, tf.float32)
            ),
            lambda: learning_rate,
        )

    lr = tf.cond(
        global_step < begin_decay_at,
        lambda: learning_rate,
        lambda: tf.train.polynomial_decay(
            learning_rate=learning_rate,
            global_step=global_step - begin_decay_at,
            decay_steps=decay_steps,
            end_learning_rate=min_lr,
            power=power,
        ),
        name='learning_rate',
    )
    return lr
