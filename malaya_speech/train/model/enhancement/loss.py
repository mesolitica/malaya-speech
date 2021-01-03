import tensorflow as tf


def snr(y_, y):
    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((y_ - y) ** 2 + 1e-6))
    sqrn_l2_norm = tf.sqrt(tf.reduce_mean(y ** 2))
    return (
        20.0
        * tf.math.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8)
        / tf.math.log(10.0)
    )


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
