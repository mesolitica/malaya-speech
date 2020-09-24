import numpy as np
from malaya_speech.model.frame import FRAME


def to_ndarray(array):

    if isinstance(array, FRAME):
        array = array.array

    if isinstance(array, list) or isinstance(array, tuple):
        array = np.array(array)
    elif isinstance(array, bytes):
        array = np.frombuffer(array, np.int16)
    return array


def to_byte(array):

    if isinstance(array, FRAME):
        array = array.array

    if isinstance(array, bytes):
        return array

    array = to_ndarray(array)
    if array.dtype == 'float':
        array = float_to_int(array)
    if array.dtype != np.int16:
        array = array.astype(np.int16)
    if not isinstance(array, bytes):
        array = array.tobytes()
    return array


def float_to_int(array, type = np.int16):

    array = to_ndarray(array)

    if array.dtype == type:
        return array

    if array.dtype not in [np.int16, np.int32, np.int64]:
        array = type(array / np.max(np.abs(array)) * np.iinfo(type).max)
    return array


def int_to_float(array, type = np.float32):

    array = to_ndarray(array)

    if array.dtype == type:
        return array

    if array.dtype not in [np.float32, np.float64]:
        array = array.astype(np.float32) / np.max(np.abs(array))

    return array
