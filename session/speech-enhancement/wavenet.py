import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import tensorflow as tf
import utils


class Config(object):
    """Configuration object that helps manage the graph."""

    def __init__(self, train_path = None):
        self.num_iters = 200000
        self.learning_rate_schedule = {
            0: 2e-4,
            90000: 4e-4 / 3,
            120000: 6e-5,
            150000: 4e-5,
            180000: 2e-5,
            210000: 6e-6,
            240000: 2e-6,
        }
        self.ae_hop_length = 512
        self.ae_bottleneck_width = 16
        self.train_path = train_path

    @staticmethod
    def _condition(x, encoding):
        """Condition the input on the encoding.
    Args:
      x: The [mb, length, channels] float tensor input.
      encoding: The [mb, encoding_length, channels] float tensor encoding.
    Returns:
      The output after broadcasting the encoding to x's shape and adding them.
    """
        mb, length, channels = x.get_shape().as_list()
        enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
        mb = tf.shape(x)[0]
        enc_mb = tf.shape(x)[0]
        enc_length = tf.shape(encoding)[1]
        length = tf.shape(x)[1]

        encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
        x = tf.reshape(x, [mb, enc_length, -1, channels])
        x += encoding
        x = tf.reshape(x, [mb, length, channels])
        #     x.set_shape([mb, length, channels])
        return x

    def build(
        self,
        inputs,
        is_training,
        rescale_inputs = True,
        include_decoder = True,
        use_reduce_mean_to_pool = False,
    ):
        """Build the graph for this configuration.
    Args:
      inputs: A dict of inputs. For training, should contain 'wav'.
      is_training: Whether we are training or not. Not used in this config.
      rescale_inputs: Whether to convert inputs to mu-law and back to unit
        scaling before passing through the model (loses gradients).
      include_decoder: bool, whether to include the decoder in the build().
      use_reduce_mean_to_pool: whether to use reduce_mean (instead of pool1d)
        for pooling.
    Returns:
      A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
      the 'quantized_input', and whatever metrics we want to track for eval.
    """
        num_stages = 10
        num_layers = 16
        filter_length = 3
        width = 512
        skip_width = 256
        ae_num_stages = 10
        ae_num_layers = 16
        ae_filter_length = 3
        ae_width = 128

        # Encode the source with 8-bit Mu-Law.
        x = inputs['inputs']
        x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)
        x = tf.expand_dims(x, 2)

        y = inputs['targets']
        y_quantized = utils.mu_law(y)
        y_scaled = tf.cast(y_quantized, tf.float32) / 128.0
        y_scaled = tf.expand_dims(y_scaled, 2)
        y = tf.expand_dims(y, 2)

        ###
        # The Non-Causal Temporal Encoder.
        ###
        en = conv1d(
            x_scaled if rescale_inputs else x,
            causal = False,
            num_filters = ae_width,
            filter_length = ae_filter_length,
            name = 'ae_startconv',
            is_training = is_training,
        )

        for num_layer in range(ae_num_layers):
            dilation = 2 ** (num_layer % ae_num_stages)
            d = tf.nn.relu(en)
            d = conv1d(
                d,
                causal = False,
                num_filters = ae_width,
                filter_length = ae_filter_length,
                dilation = dilation,
                name = 'ae_dilatedconv_%d' % (num_layer + 1),
                is_training = is_training,
            )
            d = tf.nn.relu(d)
            en += conv1d(
                d,
                num_filters = ae_width,
                filter_length = 1,
                name = 'ae_res_%d' % (num_layer + 1),
                is_training = is_training,
            )

        en = conv1d(
            en,
            num_filters = self.ae_bottleneck_width,
            filter_length = 1,
            name = 'ae_bottleneck',
            is_training = is_training,
        )

        if use_reduce_mean_to_pool:
            # Depending on the accelerator used for training, masked.pool1d may
            # lead to out of memory error.
            # reduce_mean is equivalent to masked.pool1d when the stride is the same
            # as the window length (which is the case here).
            batch_size, unused_length, depth = en.shape.as_list()
            en = tf.reshape(en, [batch_size, -1, self.ae_hop_length, depth])
            en = tf.reduce_mean(en, axis = 2)
        else:
            en = pool1d(en, self.ae_hop_length, name = 'ae_pool', mode = 'avg')
        encoding = en

        if not include_decoder:
            return {'encoding': encoding}

        ###
        # The WaveNet Decoder.
        ###
        print('WaveNet Decoder')
        l = shift_right(x_scaled if rescale_inputs else x)
        l = conv1d(
            l,
            num_filters = width,
            filter_length = filter_length,
            name = 'startconv',
            is_training = is_training,
        )

        # Set up skip connections.
        s = conv1d(
            l,
            num_filters = skip_width,
            filter_length = 1,
            name = 'skip_start',
            is_training = is_training,
        )

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2 ** (i % num_stages)
            d = conv1d(
                l,
                num_filters = 2 * width,
                filter_length = filter_length,
                dilation = dilation,
                name = 'dilatedconv_%d' % (i + 1),
                is_training = is_training,
            )
            d = self._condition(
                d,
                conv1d(
                    en,
                    num_filters = 2 * width,
                    filter_length = 1,
                    name = 'cond_map_%d' % (i + 1),
                    is_training = is_training,
                ),
            )

            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            l += conv1d(
                d,
                num_filters = width,
                filter_length = 1,
                name = 'res_%d' % (i + 1),
                is_training = is_training,
            )
            s += conv1d(
                d,
                num_filters = skip_width,
                filter_length = 1,
                name = 'skip_%d' % (i + 1),
                is_training = is_training,
            )

        s = tf.nn.relu(s)
        s = conv1d(
            s,
            num_filters = skip_width,
            filter_length = 1,
            name = 'out1',
            is_training = is_training,
        )
        s = self._condition(
            s,
            conv1d(
                en,
                num_filters = skip_width,
                filter_length = 1,
                name = 'cond_map_out1',
                is_training = is_training,
            ),
        )
        s = tf.nn.relu(s)

        ###
        # Compute the logits and get the loss.
        ###
        logits = conv1d(
            s,
            num_filters = 256,
            filter_length = 1,
            name = 'logits',
            is_training = is_training,
        )
        logits = tf.reshape(logits, [-1, 256])
        probs = tf.nn.softmax(logits, name = 'softmax')
        x_indices = tf.cast(tf.reshape(y_quantized, [-1]), tf.int32) + 128
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = logits, labels = x_indices, name = 'nll'
            ),
            0,
            name = 'loss',
        )

        return {
            'predictions': probs,
            'loss': loss,
            'eval': {'nll': loss},
            'quantized_input': x_quantized,
            'encoding': encoding,
        }


def shift_right(x):
    """Shift the input over by one and a zero to the front.

  Args:
    x: The [mb, time, channels] tensor input.

  Returns:
    x_sliced: The [mb, time, channels] tensor output.
  """
    shape = x.get_shape().as_list()
    x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
    x_sliced = tf.slice(x_padded, [0, 0, 0], tf.stack([-1, tf.shape(x)[1], -1]))
    x_sliced.set_shape(shape)
    return x_sliced


def mul_or_none(a, b):
    """Return the element wise multiplicative of the inputs.

  If either input is None, we return None.

  Args:
    a: A tensor input.
    b: Another tensor input with the same type as a.

  Returns:
    None if either input is None. Otherwise returns a * b.
  """
    if a is None or b is None or isinstance(a, tf.Tensor):
        return None
    return int(a * b)


def time_to_batch(x, block_size):
    """Splits time dimension (i.e. dimension 1) of `x` into batches.

  Within each batch element, the `k*block_size` time steps are transposed,
  so that the `k` time steps in each output batch element are offset by
  `block_size` from each other.

  The number of input time steps must be a multiple of `block_size`.

  Args:
    x: Tensor of shape [nb, k*block_size, n] for some natural number k.
    block_size: number of time steps (i.e. size of dimension 1) in the output
      tensor.

  Returns:
    Tensor of shape [nb*block_size, k, n]
  """
    # x = [b, t, 1]
    shape = x.get_shape().as_list()
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    y = tf.reshape(x, [batch_size, length // block_size, block_size, shape[2]])
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, [batch_size * block_size, length // block_size, shape[2]])
    return y


def batch_to_time(x, block_size):
    """Inverse of `time_to_batch(x, block_size)`.

  Args:
    x: Tensor of shape [nb*block_size, k, n] for some natural number k.
    block_size: number of time steps (i.e. size of dimension 1) in the output
      tensor.

  Returns:
    Tensor of shape [nb, k*block_size, n].
  """
    # x = [b, t, 1]
    shape = x.get_shape().as_list()
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    y = tf.reshape(x, [batch_size // block_size, block_size, length, shape[2]])
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, [batch_size // block_size, length * block_size, shape[2]])
    return y


def conv1d(
    x,
    num_filters,
    filter_length,
    name,
    dilation = 1,
    causal = True,
    kernel_initializer = tf.uniform_unit_scaling_initializer(1.0),
    biases_initializer = tf.constant_initializer(0.0),
    is_training = True,
):
    """Fast 1D convolution that supports causal padding and dilation.

  Args:
    x: The [mb, time, channels] float tensor that we convolve.
    num_filters: The number of filter maps in the convolution.
    filter_length: The integer length of the filter.
    name: The name of the scope for the variables.
    dilation: The amount of dilation.
    causal: Whether or not this is a causal convolution.
    kernel_initializer: The kernel initialization function.
    biases_initializer: The biases initialization function.
    is_training: Whether or not ot use traininable variables.

  Returns:
    y: The output of the 1D convolution.
  """
    batch_size, length, num_input_channels = x.get_shape().as_list()
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    kernel_shape = [1, filter_length, num_input_channels, num_filters]
    strides = [1, 1]
    biases_shape = [num_filters]
    padding = 'VALID' if causal else 'SAME'

    x_ttb = time_to_batch(x, dilation)
    if filter_length > 1 and causal:
        x_ttb = tf.pad(x_ttb, [[0, 0], [filter_length - 1, 0], [0, 0]])

    x_ttb_shape = x_ttb.get_shape().as_list()
    x_4d = tf.reshape(
        x_ttb, [tf.shape(x_ttb)[0], 1, tf.shape(x_ttb)[1], num_input_channels]
    )
    y = tf.layers.conv2d(
        x_4d,
        num_filters,
        [1, filter_length],
        strides = strides,
        padding = padding,
        kernel_initializer = kernel_initializer,
        bias_initializer = biases_initializer,
    )
    #     y = tf.nn.conv2d(x_4d, weights, strides, padding = padding)
    #     y = tf.nn.bias_add(y, biases)
    y_shape = y.get_shape().as_list()
    y = tf.reshape(y, [tf.shape(y)[0], tf.shape(y)[2], num_filters])
    y = batch_to_time(y, dilation)
    #     y.set_shape([batch_size, length, num_filters])
    return y


def pool1d(x, window_length, name, mode = 'avg', stride = None):
    """1D pooling function that supports multiple different modes.

  Args:
    x: The [mb, time, channels] float tensor that we are going to pool over.
    window_length: The amount of samples we pool over.
    name: The name of the scope for the variables.
    mode: The type of pooling, either avg or max.
    stride: The stride length.

  Returns:
    pooled: The [mb, time // stride, channels] float tensor result of pooling.
  """
    if mode == 'avg':
        pool_fn = tf.nn.avg_pool
    elif mode == 'max':
        pool_fn = tf.nn.max_pool

    stride = stride or window_length
    batch_size, length, num_channels = x.get_shape().as_list()
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    #     assert length % window_length == 0
    #     assert length % stride == 0

    window_shape = [1, 1, window_length, 1]
    strides = [1, 1, stride, 1]
    x_4d = tf.reshape(x, [batch_size, 1, length, num_channels])
    pooled = pool_fn(x_4d, window_shape, strides, padding = 'SAME', name = name)
    return tf.reshape(pooled, [batch_size, length // stride, num_channels])


import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import librosa
from glob import glob
import soundfile as sf
import random
from pysndfx import AudioEffectsChain
from scipy.special import expit
import dask.bag as db
from itertools import cycle
import mp

# files = glob('../youtube/clean-wav/*.wav')
files = glob('../speech-bahasa/LibriSpeech/*/*/*/*.flac') + glob(
    '../youtube/clean-wav/*.wav'
)
files = list(set(files))
random.shuffle(files)
print(len(files))

file_cycle = cycle(files)

import pickle

with open('../youtube/ambients.pkl', 'rb') as fopen:
    ambient = pickle.load(fopen)


def sox_reverb(
    y, reverberance = 1, hf_damping = 1, room_scale = 1, stereo_depth = 1
):
    apply_audio_effects = AudioEffectsChain().reverb(
        reverberance = reverberance,
        hf_damping = hf_damping,
        room_scale = room_scale,
        stereo_depth = stereo_depth,
        pre_delay = 20,
        wet_gain = 0,
        wet_only = False,
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def sox_augment_low(
    y,
    min_bass_gain = 5,
    reverberance = 1,
    hf_damping = 1,
    room_scale = 1,
    stereo_depth = 1,
    negate = 1,
):
    if negate:
        min_bass_gain = -min_bass_gain
    apply_audio_effects = (
        AudioEffectsChain()
        .lowshelf(gain = min_bass_gain, frequency = 300, slope = 0.1)
        .reverb(
            reverberance = reverberance,
            hf_damping = hf_damping,
            room_scale = room_scale,
            stereo_depth = stereo_depth,
            pre_delay = 20,
            wet_gain = 0,
            wet_only = False,
        )
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def sox_augment_high(
    y,
    min_bass_gain = 5,
    reverberance = 1,
    hf_damping = 1,
    room_scale = 1,
    stereo_depth = 1,
    negate = 1,
):
    if negate:
        min_bass_gain = -min_bass_gain
    apply_audio_effects = (
        AudioEffectsChain()
        .highshelf(
            gain = -min_bass_gain * (1 - expit(np.max(y))),
            frequency = 300,
            slope = 0.1,
        )
        .reverb(
            reverberance = reverberance,
            hf_damping = hf_damping,
            room_scale = room_scale,
            stereo_depth = stereo_depth,
            pre_delay = 20,
            wet_gain = 0,
            wet_only = False,
        )
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def sox_augment_combine(
    y,
    min_bass_gain_low = 5,
    min_bass_gain_high = 5,
    reverberance = 1,
    hf_damping = 1,
    room_scale = 1,
    stereo_depth = 1,
):

    apply_audio_effects = (
        AudioEffectsChain()
        .lowshelf(gain = min_bass_gain_low, frequency = 300, slope = 0.1)
        .highshelf(gain = -min_bass_gain_high, frequency = 300, slope = 0.1)
        .reverb(
            reverberance = reverberance,
            hf_damping = hf_damping,
            room_scale = room_scale,
            stereo_depth = stereo_depth,
            pre_delay = 20,
            wet_gain = 0,
            wet_only = False,
        )
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def fftnoise(f):
    f = np.array(f, dtype = 'complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def add_band_limited_noise(
    sample, min_freq = 200, max_freq = 24000, samplerate = 16000
):
    freqs = np.abs(np.fft.fftfreq(len(sample), 1 / samplerate))
    f = np.zeros(len(sample))
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return sample + fftnoise(f)


def add_uniform_noise(sample, power = 0.01):
    y_noise = sample.copy()
    noise_amp = power * np.random.uniform() * np.amax(y_noise)
    return y_noise.astype('float64') + noise_amp * np.random.normal(
        size = y_noise.shape[0]
    )


def random_sampling(sample, sr, length = 500):
    sr = int(sr / 1000)
    up = len(sample) - (sr * length)
    if up < 1:
        r = 0
    else:
        r = np.random.randint(0, up)
    return sample[r : r + sr * length]


def add_noise(sample, noise, random_sample = True, factor = 0.1):
    y_noise = sample.copy()
    if len(y_noise) > len(noise):
        noise = np.tile(noise, int(np.ceil(len(y_noise) / len(noise))))
    else:
        if random_sample:
            noise = noise[np.random.randint(0, len(noise) - len(y_noise) + 1) :]
    return y_noise + noise[: len(y_noise)] * factor


def read_flac(file):
    data, old_samplerate = sf.read(file)
    if len(data.shape) == 2:
        data = data[:, 0]
    return data, old_samplerate


def read_wav(file):
    y, sr = librosa.load(file, sr = None)
    return y, sr


def read_file(file):
    if '.flac' in file:
        y, sr = read_flac(file)
    if '.wav' in file:
        y, sr = read_wav(file)
    return y, sr


def combine_speakers(files, n = 5):
    w_samples = random.sample(files, n)
    y = [w_samples[0]]
    left = w_samples[0]
    for i in range(1, n):

        right = w_samples[i]

        overlap = random.uniform(0.01, 1.5)
        left_len = int(overlap * len(left))

        padded_right = np.pad(right, (left_len, 0))

        if len(left) > len(padded_right):
            padded_right = np.pad(
                padded_right, (0, len(left) - len(padded_right))
            )
        else:
            left = np.pad(left, (0, len(padded_right) - len(left)))

        y.append(padded_right)
        left = left + padded_right
    return left, y


def sampling(combined, frame_duration_ms = 700, sample_rate = 16000):
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    results = []
    while offset + n <= len(combined):
        results.append(combined[offset : offset + n])
        offset += n
    if offset < len(combined):
        results.append(combined[offset:])

    return results


def calc(signal):

    choice = random.randint(0, 5)
    if choice == 0:
        x = sox_augment_high(
            signal,
            min_bass_gain = random.randint(25, 50),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = 1,
        )
    if choice == 1:
        x = sox_augment_high(
            signal,
            min_bass_gain = random.randint(25, 70),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = 0,
        )
    if choice == 2:
        x = sox_augment_low(
            signal,
            min_bass_gain = random.randint(5, 30),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 50),
            negate = random.randint(0, 1),
        )
    if choice == 3:
        x = sox_augment_combine(
            signal,
            min_bass_gain_high = random.randint(25, 70),
            min_bass_gain_low = random.randint(5, 30),
            reverberance = random.randint(0, 80),
            hf_damping = 10,
            room_scale = random.randint(0, 90),
        )
    if choice == 4:
        x = sox_reverb(
            signal,
            reverberance = random.randint(10, 80),
            hf_damping = 10,
            room_scale = random.randint(10, 90),
        )
    if choice == 5:
        x = signal

    if random.randint(0, 1):
        x = add_uniform_noise(x, power = random.uniform(0.005, 0.01))

    return x


def loop(files):
    files = files[0]
    results = []
    for f in files:
        results.append(calc(f))
    return results


def generate(batch_size = 30, core = 6):
    while True:
        batch_files = [next(file_cycle) for _ in range(batch_size)]
        print(batch_files)
        print('before wavs')
        wavs = [read_file(f)[0] for f in batch_files]
        print('after wavs')

        samples = []
        print('before iterating wavs')
        for wav in wavs:
            if random.random() < 0.7:
                signal = wav.copy()

                if random.randint(0, 1):
                    signal_ = random.choice(wavs)
                    signal = add_noise(
                        signal, signal_, factor = random.uniform(0.6, 1.0)
                    )

            else:
                r = random.randint(2, 6)
                signal = combine_speakers(wavs, min(len(wavs), r))[0]

            signal = random_sampling(
                signal, 16000, random.randint(60000, 240000)
            )

            samples.append(signal)

        R = []
        for s in samples:
            if random.random() > 0.8:
                signal_ = random.choice(ambient)
                s = add_noise(s, signal_, factor = random.uniform(0.1, 0.3))
            R.append(s)

        print('len samples', len(samples))
        results = mp.multiprocessing(R, loop, cores = min(len(samples), core))
        print('after len samples', len(samples))

        X, Y = [], []
        for o in range(len(samples)):
            X.extend(sampling(results[o], 4000))
            Y.extend(sampling(samples[o], 4000))

        for o in range(len(X)):
            if (
                not (np.isnan(X[o]).any() or np.isnan(Y[o]).any())
                and np.max(X[o]) <= 1.0
                and np.min(X[o]) >= -1.0
                and np.max(Y[o]) <= 1.0
                and np.min(Y[o]) >= -1.0
            ):
                yield {'inputs': X[o], 'targets': Y[o]}


def get_dataset(batch_size = 32, shuffle_size = 128, prefetch_size = 128):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {'inputs': tf.float32, 'targets': tf.float32},
            output_shapes = {
                'inputs': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
            },
        )
        dataset = dataset.shuffle(shuffle_size)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'inputs': tf.TensorShape([64000]),
                'targets': tf.TensorShape([64000]),
            },
            padding_values = {
                'inputs': tf.constant(0, dtype = tf.float32),
                'targets': tf.constant(0, dtype = tf.float32),
            },
        )
        dataset = dataset.prefetch(prefetch_size)
        return dataset

    return get


init_lr = 1e-4
epochs = 500000


def model_fn(features, labels, mode, params):
    config = Config()
    graph = config.build(features, is_training = True)
    loss = graph['loss']
    tf.identity(loss, 'train_loss')
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(
            value = init_lr, shape = [], dtype = tf.float32
        )
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            epochs,
            end_learning_rate = 1e-6,
            power = 1.0,
            cycle = False,
        )
        #         optimizer = tf.train.RMSPropOptimizer(
        #             learning_rate, decay = 0.9, momentum = 0.9, epsilon = 1.0
        #         )
        optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate, epsilon = 1e-8
        )

        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


import malaya_speech.train as train

train_hooks = [tf.train.LoggingTensorHook(['train_loss'], every_n_iter = 1)]
train_dataset = get_dataset(batch_size = 2)

save_directory = 'output-wavenet-speech-enhancement'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 3,
    log_step = 1,
    save_checkpoint_step = 2500,
    max_steps = epochs,
    train_hooks = train_hooks,
    eval_step = 0,
)
