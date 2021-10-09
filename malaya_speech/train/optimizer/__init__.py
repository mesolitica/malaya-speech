# Copyright (c) 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.compat.v1.train import MomentumOptimizer
import tensorflow.compat.v1 as tf

from ..loss import AutomaticLossScaler
from ..utils import mask_nans, check_params
from . import adamw
from . import gradient

OPTIMIZER_CLS_NAMES = {
    'Adagrad': tf.train.AdagradOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'Ftrl': tf.train.FtrlOptimizer,
    'Momentum': tf.train.MomentumOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer,
    'SGD': tf.train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = [
    'learning_rate',
    'gradients',
    'gradient_norm',
    'global_gradient_norm',
    'variables',
    'variable_norm',
    'larc_summaries',
    'loss_scale',
]


class NovoGrad(MomentumOptimizer):
    """
  Optimizer that implements SGD with layer-wise normalized gradients,
  when normalization is done by sqrt(ema(sqr(grads))), similar to Adam
    ```
    Second moment = ema of Layer-wise sqr of grads:
       v_t <-- beta2*v_{t-1} + (1-beta2)*(g_t)^2
    First moment has two mode:
    1. moment of grads normalized by u_t:
       m_t <- beta1*m_{t-1} + lr_t * [ g_t/sqrt(v_t+epsilon)]
    1. moment similar to Adam: ema of grads normalized by u_t:
       m_t <- beta1*m_{t-1} + lr_t * [(1-beta1)*(g_t/sqrt(v_t+epsilon))]
    if weight decay add wd term after grads are rescaled by 1/sqrt(v_t):
       m_t <- beta1*m_{t-1} + lr_t * [g_t/sqrt(v_t+epsilon) + wd*w_{t-1}]
    Weight update:
       w_t <- w_{t-1} - *m_t
    ```
  """

    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.95,
        beta2=0.98,
        epsilon=1e-8,
        weight_decay=0.0,
        grad_averaging=False,
        use_locking=False,
        name='NovoGrad',
    ):
        """Constructor:
    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      beta1: A `Tensor` or a float, used in ema for momentum.Default = 0.95.
      beta2: A `Tensor` or a float, used in ema for grad norms.Default = 0.99.
      epsilon: a float.  Default = 1e-8.
      weight_decay: A `Tensor` or a float, Default = 0.0.
      grad_averaging: switch between Momentum and SAG, Default = False,
      use_locking: If `True` use locks for update operations.
      name: Optional, name prefix for the ops created when applying
        gradients.  Defaults to "NovoGrad".
      use_nesterov: If `True` use Nesterov Momentum.
    """
        super(NovoGrad, self).__init__(
            learning_rate,
            momentum=beta1,
            use_locking=use_locking,
            name=name,
            use_nesterov=False,
        )
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._wd = weight_decay
        self._grad_averaging = grad_averaging
        self._grads_ema = None

        # Tensor versions, converted to tensors in apply_gradients
        # self._beta1_t = None
        # self._beta2_t = None
        # self._wd_t = None

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        # self._beta1_t = ops.convert_to_tensor(self._beta1, name='beta1', dtype = tf.float32)
        # self._beta2_t = ops.convert_to_tensor(self._beta2, name='beta2', dtype = tf.float32)

        # init ema variables if required
        len_vars = len(grads_and_vars)
        if self._grads_ema is None:
            self._grads_ema = [None] * len_vars
            for i in range(len_vars):
                self._grads_ema[i] = tf.get_variable(
                    name='nvgrad2_ema' + str(i),
                    shape=[],
                    dtype=tf.float32,
                    initializer=tf.keras.initializers.Zeros(),
                    trainable=False,
                )

        # compute ema for grads^2 for each layer
        for i, (grad, var) in enumerate(grads_and_vars):
            g_2 = tf.reduce_sum(tf.square(x=tf.cast(grad, tf.float32)))
            self._grads_ema[i] = tf.cond(
                tf.equal(self._grads_ema[i], 0.0),
                lambda: g_2,
                lambda: self._grads_ema[i] * self._beta2
                + g_2 * (1.0 - self._beta2),
            )

            grad *= 1.0 / tf.sqrt(self._grads_ema[i] + self._epsilon)
            # weight decay
            if self._wd > 0.0:
                grad += self._wd * var
            # Momentum --> SAG
            if self._grad_averaging:
                grad *= 1.0 - self._beta1
            grads_and_vars[i] = (grad, var)

        # call Momentum to do update
        return super(NovoGrad, self).apply_gradients(
            grads_and_vars, global_step=global_step, name=name
        )


def optimize_loss(
    loss,
    optimizer,
    optimizer_params,
    learning_rate_decay_fn,
    var_list=None,
    dtype=tf.float32,
    clip_gradients=None,
    summaries=None,
    larc_params=None,
    loss_scaling=1.0,
    loss_scaling_params=None,
):

    if summaries is None:
        summaries = ['learning_rate', 'global_gradient_norm', 'loss_scale']
    else:
        for summ in summaries:
            if summ not in OPTIMIZER_SUMMARIES:
                raise ValueError(
                    'Summaries should be one of [{}], you provided {}.'.format(
                        ', '.join(OPTIMIZER_SUMMARIES), summ
                    )
                )

    if clip_gradients is not None and larc_params is not None:
        raise AttributeError(
            'LARC and gradient norm clipping should not be used together'
        )

    global_step = tf.train.get_or_create_global_step()
    lr = learning_rate_decay_fn(global_step)
    if 'learning_rate' in summaries:
        tf.summary.scalar('learning_rate', lr)

    with tf.variable_scope('Loss_Optimization'):
        update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        loss = control_flow_ops.with_dependencies(list(update_ops), loss)

        if optimizer == 'AdamW':
            optimizer_params['weight_decay'] = (
                optimizer_params['weight_decay'] * lr
            )

        if isinstance(optimizer, str):
            if optimizer not in OPTIMIZER_CLS_NAMES:
                raise ValueError(
                    'Optimizer name should be one of [{}], you provided {}.'.format(
                        ', '.join(OPTIMIZER_CLS_NAMES), optimizer
                    )
                )

        opt = optimizer(learning_rate=lr, **optimizer_params)
        if isinstance(loss_scaling, str):
            loss_scaling = AutomaticLossScaler(
                algorithm=loss_scaling, params=loss_scaling_params
            )
            if 'loss_scale' in summaries:
                tf.summary.scalar('loss_scale', loss_scaling.loss_scale)

        grads_and_vars = opt.compute_gradients(
            loss, colocate_gradients_with_ops=True, var_list=var_list
        )
        grad_updates = opt.apply_gradients(
            post_process_gradients(
                grads_and_vars,
                lr=lr,
                clip_gradients=clip_gradients,
                larc_params=larc_params,
                summaries=summaries,
            ),
            global_step=global_step,
        )
        train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)
        return train_tensor


def post_process_gradients(
    grads_and_vars, summaries, lr, clip_gradients, larc_params
):
    """Applies post processing to gradients, i.e. clipping, LARC, summaries."""
    if 'global_gradient_norm' in summaries:
        tf.summary.scalar(
            'global_gradient_norm', _global_norm_with_cast(grads_and_vars)
        )

    # Optionally clip gradients by global norm.
    if clip_gradients is not None:
        grads_and_vars = _clip_gradients_by_norm(grads_and_vars, clip_gradients)

    # Add histograms for variables, gradients and gradient norms.

    if 'global_gradient_norm' in summaries:
        for gradient, variable in grads_and_vars:
            if isinstance(gradient, tf.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient

            if isinstance(variable, tf.IndexedSlices):
                var_values = variable.values
            else:
                var_values = variable

            if grad_values is not None:
                var_name = variable.name.replace(':', '_')
                if 'gradients' in summaries:
                    # need to mask nans for automatic loss scaling
                    tf.summary.histogram(
                        'gradients/%s' % var_name, mask_nans(grad_values)
                    )
                if 'gradient_norm' in summaries:
                    tf.summary.scalar(
                        'gradient_norm/%s' % var_name, tf.norm(grad_values)
                    )
                if 'variables' in summaries:
                    tf.summary.histogram('variables/%s' % var_name, var_values)
                if 'variable_norm' in summaries:
                    tf.summary.scalar(
                        'variable_norm/%s' % var_name, tf.norm(var_values)
                    )

    if clip_gradients is not None and 'global_gradient_norm' in summaries:
        tf.summary.scalar(
            'global_clipped_gradient_norm',
            _global_norm_with_cast(grads_and_vars),
        )

    # LARC gradient re-scaling
    if larc_params is not None:
        check_params(
            config=larc_params,
            required_dict={'larc_eta': float},
            optional_dict={
                'larc_mode': ['clip', 'scale'],
                'min_update': float,
                'epsilon': float,
            },
        )
        larc_eta = larc_params['larc_eta']
        larc_mode = larc_params.get('larc_mode', 'clip')
        min_update = larc_params.get('min_update', 1e-7)
        eps = larc_params.get('epsilon', 1e-7)

        grads_and_vars_larc = [None] * len(grads_and_vars)
        for idx, (g, v) in enumerate(grads_and_vars):
            var_dtype = v.dtype
            v_norm = tf.norm(tensor=tf.cast(v, tf.float32), ord=2)
            g_norm = tf.norm(tensor=tf.cast(g, tf.float32), ord=2)

            if larc_mode == 'clip':
                larc_grad_update = tf.maximum(
                    larc_eta * v_norm / (lr * (g_norm + eps)), min_update
                )
                if 'larc_summaries' in summaries:
                    tf.summary.scalar(
                        'larc_clip_on/{}'.format(v.name),
                        tf.cast(tf.less(larc_grad_update, 1.0), tf.int32),
                    )
                larc_grad_update = tf.minimum(larc_grad_update, 1.0)
            else:
                larc_grad_update = tf.maximum(
                    larc_eta * v_norm / (g_norm + eps), min_update
                )
            larc_grad_update = tf.saturate_cast(larc_grad_update, var_dtype)
            grads_and_vars_larc[idx] = (larc_grad_update * g, v)

            # adding additional summary
            if 'larc_summaries' in summaries:
                tf.summary.scalar(
                    'larc_grad_update/{}'.format(v.name), larc_grad_update
                )
                tf.summary.scalar(
                    'larc_final_lr/{}'.format(v.name),
                    tf.cast(lr, var_dtype) * larc_grad_update,
                )
        grads_and_vars = grads_and_vars_larc
    return grads_and_vars


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    dtypes = [var.dtype for var in variables]

    # Clip gradients in float32
    clipped_gradients, _ = _clip_by_global_norm(
        gradients,
        clip_gradients,
        use_norm=_global_norm_with_cast(grads_and_vars),
    )

    # Convert gradients back to the proper dtype
    clipped_gradients = [
        tf.cast(grad, dtype) for grad, dtype in zip(clipped_gradients, dtypes)
    ]

    return list(zip(clipped_gradients, variables))


def _global_norm_with_cast(grads_and_vars):
    return tf.global_norm(
        list(
            map(lambda x: tf.cast(x, tf.float32), list(zip(*grads_and_vars))[0])
        )
    )


def _clip_by_global_norm(t_list, clip_norm, use_norm, name=None):
    """Clips values of multiple tensors by the ratio of the sum of their norms.
  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. The global
  norm is expected to be pre-computed and passed as use_norm.
  To perform the clipping, the values `t_list[i]` are set to:
      t_list[i] * clip_norm / max(global_norm, clip_norm)
  where:
      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.
  Any of the entries of `t_list` that are of type `None` are ignored.
  This is the correct way to perform gradient clipping (for example, see
  [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
  ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).
  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.
  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).
  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.
  Raises:
    TypeError: If `t_list` is not a sequence.
  """
    if not isinstance(t_list, collections.Sequence) or isinstance(
        t_list, six.string_types
    ):
        raise TypeError('t_list should be a sequence')
    t_list = list(t_list)

    # Removed as use_norm should always be passed
    # if use_norm is None:
    #   use_norm = global_norm(t_list, name)

    with tf.name_scope(
        name, 'clip_by_global_norm', t_list + [clip_norm]
    ) as name:
        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        scale = clip_norm * tf.minimum(
            1.0 / use_norm, tf.ones([1], dtype=use_norm.dtype) / clip_norm
        )

        values = [
            tf.cast(
                tf.convert_to_tensor(
                    t.values if isinstance(t, tf.IndexedSlices) else t,
                    name='t_%d' % i,
                ),
                dtype=tf.float32,
            )
            if t is not None
            else t
            for i, t in enumerate(t_list)
        ]

        values_clipped = []
        for i, v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.colocate_with(v):
                    values_clipped.append(
                        tf.identity(v * scale, name='%s_%d' % (name, i))
                    )

        list_clipped = [
            tf.IndexedSlices(c_v, t.indices, t.dense_shape)
            if isinstance(t, tf.IndexedSlices)
            else c_v
            for (c_v, t) in zip(values_clipped, t_list)
        ]

    return list_clipped, use_norm
