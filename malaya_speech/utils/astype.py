import numpy as np
from malaya_speech.model.frame import Frame


def to_ndarray(array):
    """
    Change list / tuple / bytes into np.array

    Parameters
    ----------
    array: list / tuple / bytes

    Returns
    -------
    result : np.array
    """

    if isinstance(array, Frame):
        array = array.array

    if isinstance(array, list) or isinstance(array, tuple):
        array = np.array(array)
    elif isinstance(array, bytes) or isinstance(array, bytearray):
        if isinstance(array, bytearray):
            array = bytes(array)
        array = np.frombuffer(array, np.int16)
    return array


def to_byte(array):
    """
    Change list / tuple / np.array into bytes

    Parameters
    ----------
    array: ist / tuple / np.array

    Returns
    -------
    result : bytes
    """

    if isinstance(array, Frame):
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
    """
    Change np.array float32 / float64 into np.int16

    Parameters
    ----------
    array: np.array
    type: np.int16

    Returns
    -------
    result : np.array
    """

    array = to_ndarray(array)

    if array.dtype == type:
        return array

    if array.dtype not in [np.int16, np.int32, np.int64]:
        if np.max(np.abs(array)) == 0:
            array[:] = 0
            array = type(array * np.iinfo(type).max)
        else:
            array = type(array / np.max(np.abs(array)) * np.iinfo(type).max)
    return array


def int_to_float(array, type = np.float32):
    """
    Change np.array int16 into np.float32

    Parameters
    ----------
    array: np.array
    type: np.float32

    Returns
    -------
    result : np.array
    """

    array = to_ndarray(array)

    if array.dtype == type:
        return array

    if array.dtype not in [np.float16, np.float32, np.float64]:
        if np.max(np.abs(array)) == 0:
            array = array.astype(np.float32)
            array[:] = 0
        else:
            array = array.astype(np.float32) / np.max(np.abs(array))

    return array
