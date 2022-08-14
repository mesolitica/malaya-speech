import tensorflow.compat.v1 as tf

import numpy as np


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform_tf(inputs,
                                              unnormalized_widths,
                                              unnormalized_heights,
                                              unnormalized_derivatives,
                                              inverse=False,
                                              tails=None,
                                              tail_bound=1.,
                                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    if tails is None:
        spline_fn = rational_quadratic_spline_tf
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline_tf
        spline_kwargs = {
            'tails': tails,
            'tail_bound': tail_bound
        }

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


def searchsorted_tf(bin_locations, inputs, eps=1e-6):
    bin_locations = tf.concat([
        bin_locations[..., :-1],
        tf.expand_dims(bin_locations[..., -1] + 1e-6, -1)], axis=-1)
    return tf.reduce_sum(
        tf.cast(inputs[..., None] >= bin_locations, tf.int32),
        axis=-1
    ) - 1


def unconstrained_rational_quadratic_spline_tf(inputs,
                                               unnormalized_widths,
                                               unnormalized_heights,
                                               unnormalized_derivatives,
                                               inverse=False,
                                               tails='linear',
                                               tail_bound=1.,
                                               min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                               min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                               min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = tf.zeros_like(inputs)
    logabsdet = tf.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = tf.pad(unnormalized_derivatives, [(0, 0), (0, 0), (0, 0), (1, 1)])
        constant = np.log(np.exp(1 - min_derivative) - 1)

        unnormalized_derivatives = tf.concat([
            tf.expand_dims(unnormalized_derivatives[..., 0] + constant, -1),
            unnormalized_derivatives[..., 1:-1],
            tf.expand_dims(unnormalized_derivatives[..., -1] + constant, -1),
        ], axis=-1)

        # outputs[outside_interval_mask] = inputs[outside_interval_mask]
        # logabsdet[outside_interval_mask] = 0

        outputs = tf.where(outside_interval_mask, inputs, outputs)
        tiled = tf.fill(tf.shape(logabsdet), 0.0)
        logabsdet = tf.where(outside_interval_mask, tiled, logabsdet)
    else:
        raise RuntimeError(f'{tails} tails are not implemented.')

    d = tf.shape(unnormalized_widths)[-1]
    unnormalized_widths = unnormalized_widths[tf.tile(tf.expand_dims(inside_interval_mask, -1), [1, 1, 1, d])]
    unnormalized_widths = tf.reshape(unnormalized_widths, [-1, d])

    d = tf.shape(unnormalized_heights)[-1]
    unnormalized_heights = unnormalized_heights[tf.tile(tf.expand_dims(inside_interval_mask, -1), [1, 1, 1, d])]
    unnormalized_heights = tf.reshape(unnormalized_heights, [-1, d])

    d = tf.shape(unnormalized_derivatives)[-1]
    unnormalized_derivatives = unnormalized_derivatives[tf.tile(tf.expand_dims(inside_interval_mask, -1), [1, 1, 1, d])]
    unnormalized_derivatives = tf.reshape(unnormalized_derivatives, [-1, d])

    inputs_ = inputs[inside_interval_mask]

    outputs_, logabsdet_ = rational_quadratic_spline_tf(
        inputs=inputs_,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    indices = tf.where(inside_interval_mask)
    outputs = tf.tensor_scatter_nd_update(
        outputs, indices, outputs_, name=None
    )
    logabsdet = tf.tensor_scatter_nd_update(
        logabsdet, indices, logabsdet_, name=None
    )

    return outputs, logabsdet


def rational_quadratic_spline_tf(inputs,
                                 unnormalized_widths,
                                 unnormalized_heights,
                                 unnormalized_derivatives,
                                 inverse=False,
                                 left=0., right=1., bottom=0., top=1.,
                                 min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                 min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                 min_derivative=DEFAULT_MIN_DERIVATIVE):

    num_bins = tf.shape(unnormalized_widths)[-1]

    widths = tf.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * tf.cast(num_bins, tf.float32)) * widths
    cumwidths = tf.cumsum(widths, axis=-1)
    cumwidths = tf.pad(cumwidths, [[0, 0], [1, 0]])
    cumwidths = (right - left) * cumwidths + left

    # cumwidths[..., 0] = left
    # cumwidths[..., -1] = right

    cumwidths = tf.concat([
        tf.expand_dims(tf.fill(tf.shape(cumwidths[..., 0]), left), -1),
        cumwidths[..., 1:-1],
        tf.expand_dims(tf.fill(tf.shape(cumwidths[..., -1]), right), -1),
    ], axis=-1)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + tf.math.softplus(unnormalized_derivatives)

    heights = tf.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * tf.cast(num_bins, tf.float32)) * heights
    cumheights = tf.cumsum(heights, axis=-1)
    cumheights = tf.pad(cumheights, [[0, 0], [1, 0]])
    cumheights = (top - bottom) * cumheights + bottom

    # cumheights[..., 0] = bottom
    # cumheights[..., -1] = top

    cumheights = tf.concat([
        tf.expand_dims(tf.fill(tf.shape(cumheights[..., 0]), bottom), -1),
        cumheights[..., 1:-1],
        tf.expand_dims(tf.fill(tf.shape(cumheights[..., -1]), top), -1),
    ], axis=-1)

    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted_tf(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted_tf(cumwidths, inputs)[..., None]

    input_cumwidths = tf.gather_nd(cumwidths, bin_idx, batch_dims=1)
    input_bin_widths = tf.gather_nd(widths, bin_idx, batch_dims=1)

    input_cumheights = tf.gather_nd(cumheights, bin_idx, batch_dims=1)
    delta = heights / widths
    input_delta = tf.gather_nd(delta, bin_idx, batch_dims=1)

    input_derivatives = tf.gather_nd(derivatives, bin_idx, batch_dims=1)
    input_derivatives_plus_one = tf.gather_nd(derivatives[..., 1:], bin_idx, batch_dims=1)

    input_heights = tf.gather_nd(heights, bin_idx, batch_dims=1)

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = (b ** 2) - 4 * a * c

        root = (2 * c) / (-b - tf.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = (input_delta ** 2) * (input_derivatives_plus_one * (root**2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * ((1 - root)**2))
        logabsdet = tf.log(derivative_numerator) - 2 * tf.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * (theta**2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = (input_delta**2) * (input_derivatives_plus_one * (theta**2)
                                                   + 2 * input_delta * theta_one_minus_theta
                                                   + input_derivatives * ((1 - theta)**2))
        logabsdet = tf.log(derivative_numerator) - 2 * tf.log(denominator)

        return outputs, logabsdet
