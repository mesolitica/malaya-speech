import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')

import malaya_speech
import malaya_speech.config
import malaya_speech.augmentation.waveform as augmentation
from malaya_speech.train.model import srgan
import numpy as np
from glob import glob
import random
from multiprocessing import Pool
from itertools import cycle
import itertools
import tensorflow as tf

np.seterr(all='raise')


def chunks(l, n):
    for i in range(0, len(l), n):
        yield (l[i: i + n], i // n)


def multiprocessing(strings, function, cores=6, returned=True):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    print('initiate pool map')
    pooled = pool.map(function, df_split)
    print('gather from pool')
    pool.close()
    pool.join()
    print('closed pool')

    if returned:
        return list(itertools.chain(*pooled))


files = glob('../youtube/clean-wav/*.wav')
random.shuffle(files)
file_cycle = cycle(files)

sr = 44100
partition_size = 2048
reduction_factor = 4


def read_wav(f):
    return malaya_speech.load(f, sr=sr)


def random_sampling(s, length):
    return augmentation.random_sampling(s, sr=sr, length=length)


def downsample(y, sr, down_sr):
    y_ = malaya_speech.resample(y, sr, down_sr)
    return malaya_speech.resample(y_, down_sr, sr)


def parallel(f):
    y = read_wav(f)[0]
    y = random_sampling(y, length=3000)
    y_ = malaya_speech.resample(y, sr, sr // reduction_factor)
    return y_, y


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(parallel(f))
    return results


def generate(batch_size=10, repeat=5):
    while True:
        fs = [next(file_cycle) for _ in range(batch_size)]
        results = multiprocessing(fs, loop, cores=len(fs))
        for _ in range(repeat):
            random.shuffle(results)
            for r in results:
                if not np.isnan(r[0]).any() and not np.isnan(r[1]).any():
                    yield {'combined': r[0], 'y': r[1]}


dataset = tf.data.Dataset.from_generator(
    generate,
    {'combined': tf.float32, 'y': tf.float32},
    output_shapes={
        'combined': tf.TensorShape([None]),
        'y': tf.TensorShape([None]),
    },
)
features = dataset.make_one_shot_iterator().get_next()

x = tf.expand_dims(features['combined'], -1)
y = tf.expand_dims(features['y'], -1)
partitioned_x = malaya_speech.tf_featurization.pad_and_partition(
    x, partition_size // reduction_factor
)
partitioned_y = malaya_speech.tf_featurization.pad_and_partition(
    y, partition_size
)

with tf.variable_scope('generator') as gen:
    generator = srgan.Model_Keras(
        partition_size // reduction_factor, num_filters=512, training=True
    )

with tf.variable_scope('discriminator') as dis:
    discriminator = srgan.MultiScaleDiscriminator(
        partition_size, num_filters=24, training=True
    )

mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()

y_hat = generator.model(partitioned_x)
y_hat.set_shape((None, partition_size, 1))
p_hat = discriminator.model(y_hat)
p = discriminator.model(partitioned_y)
adv_loss = 0.0
for i in range(len(p_hat)):
    adv_loss += mae_loss(tf.ones_like(p_hat[i]), p_hat[i])
adv_loss /= i + 1
generator_loss = adv_loss

tf.summary.scalar('adv_loss', adv_loss)

y_hat = generator.model(partitioned_x)
y_hat.set_shape((None, partition_size, 1))
p_hat = discriminator.model(y_hat)
p = discriminator.model(partitioned_y)
real_loss = 0.0
fake_loss = 0.0
for i in range(len(p)):
    real_loss += mse_loss(tf.ones_like(p[i]), p[i])
    fake_loss += mse_loss(tf.zeros_like(p[i]), p[i])

real_loss /= i + 1
fake_loss /= i + 1
dis_loss = real_loss + fake_loss
discriminator_loss = dis_loss

tf.summary.scalar('real_loss', real_loss)
tf.summary.scalar('fake_loss', fake_loss)

summaries = tf.summary.merge_all()

t_vars = tf.trainable_variables()
print(t_vars)
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('generator')]

d_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9).minimize(
    discriminator_loss, var_list=d_vars
)
g_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9).minimize(
    generator_loss, var_list=g_vars
)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

checkpoint = 5000
write_tensorboard = 100
epoch = 1_000_000
path = 'srgan-512-multiscale'

writer = tf.summary.FileWriter(f'./{path}')

ckpt_path = tf.train.latest_checkpoint(path)
if ckpt_path:
    saver.restore(sess, ckpt_path)

for i in range(epoch):
    g_loss, _ = sess.run([generator_loss, g_optimizer])
    d_loss, _ = sess.run([discriminator_loss, d_optimizer])
    s = sess.run(summaries)

    if i % checkpoint == 0:
        saver.save(sess, f'{path}/model.ckpt', global_step=i)

    if i % write_tensorboard == 0:
        writer.add_summary(s, i)

    print(i, g_loss, d_loss)
