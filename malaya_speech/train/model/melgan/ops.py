from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops.nn_ops import conv1d


class _NonAtrousConvolution(object):
    """Helper class for _non_atrous_convolution.
    Note that this class assumes that shapes of input and filter passed to
    __call__ are compatible with input_shape and filter_shape passed to the
    constructor.
    Arguments:
      input_shape: static input shape, i.e. input.get_shape().
      filter_shape: static filter shape, i.e. filter.get_shape().
      padding: see _non_atrous_convolution.
      data_format: see _non_atrous_convolution.
      strides: see _non_atrous_convolution.
      name: see _non_atrous_convolution.
    """

    def __init__(
            self,
            input_shape,
            filter_shape,  # pylint: disable=redefined-builtin
            padding,
            data_format=None,
            strides=None,
            name=None):
        filter_shape = filter_shape.with_rank(input_shape.ndims)
        self.padding = padding
        self.name = name
        input_shape = input_shape.with_rank(filter_shape.ndims)
        if input_shape.ndims is None:
            raise ValueError("Rank of convolution must be known")
        if input_shape.ndims < 3 or input_shape.ndims > 5:
            raise ValueError(
                "`input` and `filter` must have rank at least 3 and at most 5")
        conv_dims = input_shape.ndims - 2
        if strides is None:
            strides = [1] * conv_dims
        elif len(strides) != conv_dims:
            raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                                    conv_dims))
        if conv_dims == 1:
            # conv1d uses the 2-d data format names
            if data_format is None:
                data_format = "NWC"
            elif data_format not in {"NCW", "NWC", "NCHW", "NHWC"}:
                raise ValueError("data_format must be \"NWC\" or \"NCW\".")
            self.strides = strides[0]
            self.data_format = data_format
            self.conv_op = self._conv1d
        elif conv_dims == 2:
            if data_format is None or data_format == "NHWC":
                data_format = "NHWC"
                strides = [1] + list(strides) + [1]
            elif data_format == "NCHW":
                strides = [1, 1] + list(strides)
            else:
                raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
            self.strides = strides
            self.data_format = data_format
            self.conv_op = conv2d
        elif conv_dims == 3:
            if data_format is None or data_format == "NDHWC":
                strides = [1] + list(strides) + [1]
            elif data_format == "NCDHW":
                strides = [1, 1] + list(strides)
            else:
                raise ValueError("data_format must be \"NDHWC\" or \"NCDHW\". Have: %s"
                                 % data_format)
            self.strides = strides
            self.data_format = data_format
            self.conv_op = gen_nn_ops.conv3d

    # Note that we need this adapter since argument names for conv1d don't match
    # those for gen_nn_ops.conv2d and gen_nn_ops.conv3d.
    # pylint: disable=redefined-builtin
    def _conv1d(self, input, filter, strides, padding, data_format, name):
        return conv1d(
            value=input,
            filters=filter,
            stride=strides,
            padding=padding,
            data_format=data_format,
            name=name)

    # pylint: enable=redefined-builtin

    def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
        return self.conv_op(
            input=inp,
            filter=filter,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            name=self.name)
