# Copyright (c) 2018 NVIDIA Corporation

import copy
import tensorflow as tf


def mp_regularizer_wrapper(regularizer):
    def func_wrapper(weights):
        if weights.dtype.base_dtype == tf.float16:
            tf.add_to_collection(
                'REGULARIZATION_FUNCTIONS', (weights, regularizer)
            )
            # disabling the inner regularizer
            return None
        return regularizer(weights)

    return func_wrapper


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


def cast_types(input_dict, dtype):
    cast_input_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, tf.Tensor):
            if value.dtype == tf.float16 or value.dtype == tf.float32:
                if value.dtype.base_dtype != dtype.base_dtype:
                    cast_input_dict[key] = tf.cast(value, dtype)
                    continue
        if isinstance(value, dict):
            cast_input_dict[key] = cast_types(input_dict[key], dtype)
            continue
        if isinstance(value, list):
            cur_list = []
            for nest_value in value:
                if isinstance(nest_value, tf.Tensor):
                    if (
                        nest_value.dtype == tf.float16
                        or nest_value.dtype == tf.float32
                    ):
                        if nest_value.dtype.base_dtype != dtype.base_dtype:
                            cur_list.append(tf.cast(nest_value, dtype))
                            continue
                cur_list.append(nest_value)
            cast_input_dict[key] = cur_list
            continue
        cast_input_dict[key] = input_dict[key]
    return cast_input_dict


class Encoder:
    @staticmethod
    def get_required_params():
        return {}

    @staticmethod
    def get_optional_params():
        return {
            'regularizer': None,
            'regularizer_params': dict,
            'initializer': None,
            'initializer_params': dict,
            'dtype': [tf.float32, tf.float16, 'mixed'],
        }

    def __init__(self, params, model, name='encoder', mode='train'):
        check_params(
            params, self.get_required_params(), self.get_optional_params()
        )
        self._params = copy.deepcopy(params)
        self._model = model

        if 'dtype' not in self._params:
            if self._model:
                self._params['dtype'] = self._model.params['dtype']
            else:
                self._params['dtype'] = tf.float32

        self._name = name
        self._mode = mode
        self._compiled = False

    def encode(self, input_dict):
        if not self._compiled:
            if 'regularizer' not in self._params:
                if self._model and 'regularizer' in self._model.params:
                    self._params['regularizer'] = copy.deepcopy(
                        self._model.params['regularizer']
                    )
                    self._params['regularizer_params'] = copy.deepcopy(
                        self._model.params['regularizer_params']
                    )

            if 'regularizer' in self._params:
                init_dict = self._params.get('regularizer_params', {})
                if self._params['regularizer'] is not None:
                    self._params['regularizer'] = self._params['regularizer'](
                        **init_dict
                    )

            if self._params['dtype'] == 'mixed':
                self._params['dtype'] = tf.float16

        if 'initializer' in self.params:
            init_dict = self.params.get('initializer_params', {})
            initializer = self.params['initializer'](**init_dict)
        else:
            initializer = None

        self._compiled = True

        with tf.variable_scope(
            self._name, initializer=initializer, dtype=self.params['dtype']
        ):
            return self._encode(self._cast_types(input_dict))

    def _cast_types(self, input_dict):
        return cast_types(input_dict, self.params['dtype'])

    def _encode(self, input_dict):
        pass

    @property
    def params(self):
        return self._params

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return self._name


class Decoder:
    """Abstract class from which all decoders must inherit.
  """

    @staticmethod
    def get_required_params():
        """Static method with description of required parameters.
      Returns:
        dict:
            Dictionary containing all the parameters that **have to** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
        return {}

    @staticmethod
    def get_optional_params():
        """Static method with description of optional parameters.
      Returns:
        dict:
            Dictionary containing all the parameters that **can** be
            included into the ``params`` parameter of the
            class :meth:`__init__` method.
    """
        return {
            'regularizer': None,  # any valid TensorFlow regularizer
            'regularizer_params': dict,
            'initializer': None,  # any valid TensorFlow initializer
            'initializer_params': dict,
            'dtype': [tf.float32, tf.float16, 'mixed'],
        }

    def __init__(self, params, model, name='decoder', mode='train'):
        """Decoder constructor.
    Note that decoder constructors should not modify TensorFlow graph, all
    graph construction should happen in the :meth:`self._decode() <_decode>`
    method.
    Args:
      params (dict): parameters describing the decoder.
          All supported parameters are listed in :meth:`get_required_params`,
          :meth:`get_optional_params` functions.
      model (instance of a class derived from :class:`Model<models.model.Model>`):
          parent model that created this decoder.
          Could be None if no model access is required for the use case.
      name (str): name for decoder variable scope.
      mode (str): mode decoder is going to be run in.
          Could be "train", "eval" or "infer".
    Config parameters:
    * **initializer** --- any valid TensorFlow initializer. If no initializer
      is provided, model initializer will be used.
    * **initializer_params** (dict) --- dictionary that will be passed to
      initializer ``__init__`` method.
    * **regularizer** --- and valid TensorFlow regularizer. If no regularizer
      is provided, model regularizer will be used.
    * **regularizer_params** (dict) --- dictionary that will be passed to
      regularizer ``__init__`` method.
    * **dtype** --- model dtype. Could be either ``tf.float16``, ``tf.float32``
      or "mixed". For details see
      :ref:`mixed precision training <mixed_precision>` section in docs. If no
      dtype is provided, model dtype will be used.
    """
        check_params(
            params, self.get_required_params(), self.get_optional_params()
        )
        self._params = copy.deepcopy(params)
        self._model = model

        if 'dtype' not in self._params:
            if self._model:
                self._params['dtype'] = self._model.params['dtype']
            else:
                self._params['dtype'] = tf.float32

        self._name = name
        self._mode = mode
        self._compiled = False

    def decode(self, input_dict):
        """Wrapper around :meth:`self._decode() <_decode>` method.
    Here name, initializer and dtype are set in the variable scope and then
    :meth:`self._decode() <_decode>` method is called.
    Args:
      input_dict (dict): see :meth:`self._decode() <_decode>` docs.
    Returns:
      see :meth:`self._decode() <_decode>` docs.
    """
        if not self._compiled:
            if 'regularizer' not in self._params:
                if self._model and 'regularizer' in self._model.params:
                    self._params['regularizer'] = copy.deepcopy(
                        self._model.params['regularizer']
                    )
                    self._params['regularizer_params'] = copy.deepcopy(
                        self._model.params['regularizer_params']
                    )

            if 'regularizer' in self._params:
                init_dict = self._params.get('regularizer_params', {})
                if self._params['regularizer'] is not None:
                    self._params['regularizer'] = self._params['regularizer'](
                        **init_dict
                    )
                if self._params['dtype'] == 'mixed':
                    self._params['regularizer'] = mp_regularizer_wrapper(
                        self._params['regularizer']
                    )

            if self._params['dtype'] == 'mixed':
                self._params['dtype'] = tf.float16

        if 'initializer' in self.params:
            init_dict = self.params.get('initializer_params', {})
            initializer = self.params['initializer'](**init_dict)
        else:
            initializer = None

        self._compiled = True

        with tf.variable_scope(
            self._name, initializer=initializer, dtype=self.params['dtype']
        ):
            return self._decode(self._cast_types(input_dict))

    def _cast_types(self, input_dict):
        """This function performs automatic cast of all inputs to decoder dtype.
    Args:
      input_dict (dict): dictionary passed to :meth:`self._decode() <_decode>`
          method.
    Returns:
      dict: same as input_dict, but with all Tensors cast to decoder dtype.
    """
        return cast_types(input_dict, self.params['dtype'])

    def _decode(self, input_dict):
        """This is the main function which should construct decoder graph.
    Typically, decoder will take hidden representation from encoder as an input
    and produce some output sequence as an output.
    Args:
      input_dict (dict): dictionary containing decoder inputs.
          If the decoder is used with :class:`models.encoder_decoder` class,
          ``input_dict`` will have the following content::
            {
              "encoder_output": dictionary returned from encoder.encode() method
              "target_tensors": data_layer.input_tensors['target_tensors']
            }
    Returns:
      dict: dictionary of decoder outputs. Typically this will be just::
          {
            "logits": logits that will be passed to Loss
            "outputs": list with actual decoded outputs, e.g. characters
                       instead of logits
          }
    """
        pass

    @property
    def params(self):
        """Parameters used to construct the decoder (dictionary)"""
        return self._params

    @property
    def mode(self):
        """Mode decoder is run in."""
        return self._mode

    @property
    def name(self):
        """Decoder name."""
        return self._name
