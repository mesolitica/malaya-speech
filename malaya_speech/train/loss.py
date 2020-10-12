import tensorflow as tf


def calculate_3d_loss(y_gt, y_pred, loss_fn):
    """Calculate 3d loss, normally it's mel-spectrogram loss."""
    y_gt_T = tf.shape(y_gt)[1]
    y_pred_T = tf.shape(y_pred)[1]

    def f1():
        return tf.slice(y_gt, [0, 0, 0], [-1, y_pred_T, -1])

    def f2():
        return y_gt

    y_gt = tf.cond(tf.greater(y_gt_T, y_pred_T), f1, f2)

    def f3():
        return tf.slice(y_pred, [0, 0, 0], [-1, y_gt_T, -1])

    def f4():
        return y_pred

    y_pred = tf.cond(tf.greater(y_pred_T, y_gt_T), f3, f4)

    # there is a mismath length when training multiple GPU.
    # we need slice the longer tensor to make sure the loss
    # calculated correctly.
    # if y_gt_T > y_pred_T:
    #     y_gt = tf.slice(y_gt, [0, 0, 0], [-1, y_pred_T, -1])
    # elif y_pred_T > y_gt_T:
    #     y_pred = tf.slice(y_pred, [0, 0, 0], [-1, y_gt_T, -1])

    loss = loss_fn(y_gt, y_pred)
    # if isinstance(loss, tuple) is False:
    #     loss = tf.reduce_mean(
    #         loss, list(range(1, len(loss.shape)))
    #     )  # shape = [B]
    # else:
    #     loss = list(loss)
    #     for i in range(len(loss)):
    #         loss[i] = tf.reduce_mean(
    #             loss[i], list(range(1, len(loss[i].shape)))
    #         )  # shape = [B]
    return loss


def calculate_2d_loss(y_gt, y_pred, loss_fn):
    """Calculate 2d loss, normally it's durrations/f0s/energys loss."""
    y_gt_T = tf.shape(y_gt)[1]
    y_pred_T = tf.shape(y_pred)[1]

    # there is a mismath length when training multiple GPU.
    # we need slice the longer tensor to make sure the loss
    # calculated correctly.
    # if
    # if y_gt_T > y_pred_T:
    #     y_gt = tf.slice(y_gt, [0, 0], [-1, y_pred_T])
    # elif y_pred_T > y_gt_T:
    #     y_pred = tf.slice(y_pred, [0, 0], [-1, y_gt_T])

    def f1():
        return tf.slice(y_gt, [0, 0], [-1, y_pred_T])

    def f2():
        return y_gt

    y_gt = tf.cond(tf.greater(y_gt_T, y_pred_T), f1, f2)

    def f3():
        return tf.slice(y_pred, [0, 0], [-1, y_gt_T])

    def f4():
        return y_pred

    y_pred = tf.cond(tf.greater(y_pred_T, y_gt_T), f3, f4)

    loss = loss_fn(y_gt, y_pred)

    return loss
