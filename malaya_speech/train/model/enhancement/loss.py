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


def weighted_sdr(noisy_speech, pred_speech, true_speech):
    pred_noise = noisy_speech - pred_speech
    true_noise = noisy_speech - true_speech
    alpha = tf.reduce_mean(tf.square(true_speech)) / tf.reduce_mean(
        tf.square(true_speech) + tf.square(noisy_speech - pred_speech)
    )
    sound_SDR = sdr(pred_speech, true_speech)
    noise_SDR = sdr(pred_noise, true_noise)
    return alpha * sound_SDR + (1 - alpha) * noise_SDR
