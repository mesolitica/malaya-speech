import tensorflow as tf
import collections
from . import discriminator

EPS = 1e-12

Model = collections.namedtuple(
    'Model',
    'outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train, global_step',
)


def create_model(
    generator,
    inputs,
    targets,
    gan_weight = 1.0,
    l1_weight = 100.0,
    lr = 0.0002,
    beta1 = 0.5,
    **kwargs
):
    with tf.variable_scope('generator'):
        outputs = generator(inputs)

    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator'):
            predict_real = discriminator.Discriminator(inputs, targets).logits

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse = True):
            predict_fake = discriminator.Discriminator(inputs, outputs).logits

    with tf.name_scope('discriminator_loss'):
        discrim_loss = tf.reduce_mean(
            -(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS))
        )

    with tf.name_scope('generator_loss'):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.name_scope('discriminator_train'):
        discrim_tvars = [
            var
            for var in tf.trainable_variables()
            if var.name.startswith('discriminator')
        ]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(
            discrim_loss, var_list = discrim_tvars
        )
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope('generator_train'):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [
                var
                for var in tf.trainable_variables()
                if var.name.startswith('generator')
            ]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(
                gen_loss, var_list = gen_tvars
            )
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay = 0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real = predict_real,
        predict_fake = predict_fake,
        discrim_loss = ema.average(discrim_loss),
        discrim_grads_and_vars = discrim_grads_and_vars,
        gen_loss_GAN = ema.average(gen_loss_GAN),
        gen_loss_L1 = ema.average(gen_loss_L1),
        gen_grads_and_vars = gen_grads_and_vars,
        outputs = outputs,
        train = tf.group(update_losses, incr_global_step, gen_train),
        global_step = global_step,
    )


def get_session(generator, inputs, targets, **kwargs):

    sess = tf.InteractiveSession()
    model = create_model(generator, inputs, targets)

    tf.summary.scalar('discriminator_loss', model.discrim_loss)
    tf.summary.scalar('generator_loss_GAN', model.gen_loss_GAN)
    tf.summary.scalar('generator_loss_L1', model.gen_loss_L1)

    summaries = tf.summary.merge_all()

    with tf.name_scope('parameter_count'):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()]
        )
    fetches = {
        'train': model.train,
        'global_step': model.global_step,
        'discrim_loss': model.discrim_loss,
        'gen_loss_GAN': model.gen_loss_GAN,
        'gen_loss_L1': model.gen_loss_L1,
    }
    fetches['discrim_loss'] = model.discrim_loss
    fetches['gen_loss_GAN'] = model.gen_loss_GAN
    fetches['gen_loss_L1'] = model.gen_loss_L1
    fetches['summary'] = summaries

    sess.run(tf.global_variables_initializer())

    return sess, fetches
