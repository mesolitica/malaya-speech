import tensorflow as tf
from tensorflow.signal import stft, inverse_stft, hann_window

separation_exponent = 2
EPSILON = 1e-10


def pad_and_partition(tensor, segment_len):
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(
        tensor, [[0, pad_size]] + [[0, 0]] * (len(tensor.shape) - 1)
    )
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(
        padded,
        tf.concat([[split, segment_len], tf.shape(padded)[1:]], axis = 0),
    )


def pad_and_reshape(
    instr_spec, frame_length, frame_step = 1024, T = 512, F = 1024
):
    spec_shape = tf.shape(instr_spec)
    extension_row = tf.zeros((spec_shape[0], spec_shape[1], 1, spec_shape[-1]))
    n_extra_row = (frame_length) // 2 + 1 - F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    extended_spec = tf.concat([instr_spec, extension], axis = 2)
    old_shape = tf.shape(extended_spec)
    new_shape = tf.concat(
        [[old_shape[0] * old_shape[1]], old_shape[2:]], axis = 0
    )
    processed_instr_spec = tf.reshape(extended_spec, new_shape)
    return processed_instr_spec


def extend_mask(
    mask,
    extension = 'zeros',
    frame_length = 4096,
    frame_step = 1024,
    T = 512,
    F = 1024,
):
    if extension == 'average':
        extension_row = tf.reduce_mean(mask, axis = 2, keepdims = True)
    elif extension == 'zeros':
        mask_shape = tf.shape(mask)
        extension_row = tf.zeros(
            (mask_shape[0], mask_shape[1], 1, mask_shape[-1])
        )
    else:
        raise ValueError(f'Invalid mask_extension parameter {extension}')
    n_extra_row = frame_length // 2 + 1 - F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    return tf.concat([mask, extension], axis = 2)


def get_stft(
    y,
    return_magnitude = True,
    frame_length = 4096,
    frame_step = 1024,
    T = 512,
    F = 1024,
):

    waveform = tf.concat(
        [tf.zeros((frame_length, 1)), tf.expand_dims(y, -1)], 0
    )
    stft_feature = tf.transpose(
        stft(
            tf.transpose(waveform),
            frame_length,
            frame_step,
            window_fn = lambda frame_length, dtype: (
                hann_window(frame_length, periodic = True, dtype = dtype)
            ),
            pad_end = True,
        ),
        perm = [1, 2, 0],
    )
    if return_magnitude:
        D = tf.abs(pad_and_partition(stft_feature, T))[:, :, :F, :]
        return stft_feature, D
    else:
        return stft_feature


def istft(
    stft_t,
    y,
    time_crop = None,
    factor = 2 / 3,
    frame_length = 4096,
    frame_step = 1024,
    T = 512,
    F = 1024,
):

    inversed = (
        inverse_stft(
            tf.transpose(stft_t, perm = [2, 0, 1]),
            frame_length,
            frame_step,
            window_fn = lambda frame_length, dtype: (
                hann_window(frame_length, periodic = True, dtype = dtype)
            ),
        )
        * factor
    )
    reshaped = tf.transpose(inversed)
    if time_crop is None:
        time_crop = tf.shape(y)[0]
    return reshaped[frame_length : frame_length + time_crop, :]
