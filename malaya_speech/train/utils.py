import tensorflow as tf

# https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/utils/utils.py#L403


def check_params(config, required_dict, optional_dict):
    if required_dict is None or optional_dict is None:
        return

    for pm, vals in required_dict.items():
        if pm not in config:
            raise ValueError('{} parameter has to be specified'.format(pm))
        else:
            if vals == str:
                vals = string_types
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError('{} has to be one of {}'.format(pm, vals))
            if (
                vals
                and not isinstance(vals, list)
                and not isinstance(config[pm], vals)
            ):
                raise ValueError('{} has to be of type {}'.format(pm, vals))

    for pm, vals in optional_dict.items():
        if vals == str:
            vals = string_types
        if pm in config:
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError('{} has to be one of {}'.format(pm, vals))
            if (
                vals
                and not isinstance(vals, list)
                and not isinstance(config[pm], vals)
            ):
                raise ValueError('{} has to be of type {}'.format(pm, vals))

    for pm in config:
        if pm not in required_dict and pm not in optional_dict:
            raise ValueError('Unknown parameter: {}'.format(pm))


def mask_nans(x):
    x_zeros = tf.zeros_like(x)
    x_mask = tf.is_finite(x)
    y = tf.where(x_mask, x, x_zeros)
    return y
