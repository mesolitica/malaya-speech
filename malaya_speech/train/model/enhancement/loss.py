import tensorflow as tf


def snr(y_, y):
    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((y_ - y) ** 2 + 1e-6, axis = [1, 2]))
    sqrn_l2_norm = tf.sqrt(tf.reduce_mean(y ** 2, axis = [1, 2]))
    snr = 20 * tf.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.log(10.0)
    avg_sqrt_l2_loss = tf.reduce_mean(sqrt_l2_loss, axis = 0)
    avg_snr = tf.reduce_mean(snr, axis = 0)
    return avg_sqrt_l2_loss, avg_snr


def sdr(pred, true, eps = 1e-8):
    return -tf.reduce_mean(true * pred) / (
        tf.reduce_mean(tf.norm(tensor = pred) * tf.norm(tensor = true)) + eps
    )


def sdr_v2(s_sep, s_ref, eps = 1e-8):
    # pred = [B, T], true = [B, T]
    s_sep = tf.expand_dims(s_sep, 1)
    s_ref = tf.expand_dims(s_ref, 1)
    s_sep_mean = tf.reduce_mean(s_sep, axis = -1, keep_dims = True)
    s_ref_mean = tf.reduce_mean(s_ref, axis = -1, keep_dims = True)
    s_sep -= s_sep_mean
    s_ref -= s_ref_mean
    s_dot = tf.reduce_sum(
        s_ref * s_sep, axis = -1, keep_dims = True
    )  # [batch_size, C(speakers), 1]
    p_ref = tf.reduce_sum(
        s_ref ** 2, axis = -1, keep_dims = True
    )  # [batch_size, C(speakers), 1]
    s_target = (
        s_dot * s_ref / (p_ref + eps)
    )  # [batch_size, C(speakers), length]
    e_noise = s_sep - s_target  # [batch_size, C(speakers), length]
    s_target_norm = tf.reduce_sum(
        s_target ** 2, axis = -1
    )  # [batch_size, C(speakers)]
    e_noise_norm = tf.reduce_sum(
        e_noise ** 2, axis = -1
    )  # [batch_size, C(speakers)]
    si_snr = 10 * tf.log(
        s_target_norm / (e_noise_norm + eps)
    )  # [batch_size, C(speakers)]
    si_snr = tf.reduce_mean(si_snr) / tf.log(10.0)
    return si_snr


def weighted_sdr(noisy_speech, pred_speech, true_speech):
    pred_noise = noisy_speech - pred_speech
    true_noise = noisy_speech - true_speech
    alpha = tf.reduce_mean(tf.square(true_speech)) / tf.reduce_mean(
        tf.square(true_speech) + tf.square(noisy_speech - pred_speech)
    )
    sound_SDR = sdr(pred_speech, true_speech)
    noise_SDR = sdr(pred_noise, true_noise)
    return alpha * sound_SDR + (1 - alpha) * noise_SDR
