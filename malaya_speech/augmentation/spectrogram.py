import tensorflow as tf

# https://github.com/Kyubyong/specAugment/blob/master/USER_DIR/speech_recognition.py
def time_warp(mel_fbanks, W = 80):
    """
    mel_fbanks: melspectrogram tensor.
        (1, timesteps or n or τ, number of mel freq bins or v, 1)
    W: int. time warp parameter.
    """
    fbank_size = tf.shape(mel_fbanks)
    n, v = fbank_size[1], fbank_size[2]

    pt = tf.random_uniform([], W, n - W, tf.int32)
    src_ctr_pt_freq = tf.range(v // 2)
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.to_float(src_ctr_pts)

    w = tf.random_uniform([], -W, W, tf.int32)
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.to_float(dest_ctr_pts)

    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)

    warped_image, _ = tf.contrib.image.sparse_image_warp(
        mel_fbanks, source_control_point_locations, dest_control_point_locations
    )
    return warped_image


def freq_mask(mel_fbanks, F = 0.1):
    """
    mel_fbanks: melspectrogram tensor.
        (1, timesteps or n, number of mel freq bins or v, 1)
    F: int. freqeuncy mask parameter.
    """
    fbank_size = tf.shape(mel_fbanks)
    n, v = fbank_size[1], fbank_size[2]

    f = tf.random_uniform(
        [], 0, tf.cast(tf.cast(v, tf.float32) * F, tf.int32), tf.int32
    )
    f0 = tf.random_uniform([], 0, v - f, tf.int32)
    mask = tf.concat(
        (
            tf.ones(shape = (1, n, v - f0 - f, 1)),
            tf.zeros(shape = (1, n, f, 1)),
            tf.ones(shape = (1, n, f0, 1)),
        ),
        2,
    )
    masked = mel_fbanks * mask
    return tf.to_float(masked)


def time_mask(mel_fbanks, T = 0.1):
    """
    mel_fbanks: melspectrogram tensor.
        (1, timesteps or n, number of mel freq bins or v, 1)
    T: int. time mask parameter.
    """
    fbank_size = tf.shape(mel_fbanks)
    n, v = fbank_size[1], fbank_size[2]
    T = tf.cast(tf.cast(n, tf.float32) * T, tf.int32)

    t = tf.random_uniform([], 0, T, tf.int32)
    t0 = tf.random_uniform([], 0, n - T, tf.int32)
    mask = tf.concat(
        (
            tf.ones(shape = (1, n - t0 - t, v, 1)),
            tf.zeros(shape = (1, t, v, 1)),
            tf.ones(shape = (1, t0, v, 1)),
        ),
        1,
    )
    masked = mel_fbanks * mask
    return tf.to_float(masked)
