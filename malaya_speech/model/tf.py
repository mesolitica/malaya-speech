import tensorflow as tf
import numpy as np
from malaya_speech.model.frame import FRAME
from malaya_speech.utils.padding import padding_sequence_nd


class CLASSIFICATION:
    def __init__(self, X, logits, vectorizer, sess, model, extra, name):
        self._X = X
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._model = model
        self._extra = extra
        self.__name__ = name

    def predict(self, inputs):
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        inputs = padding_sequence_nd(inputs)

        if 'vggvox' in self._model:
            inputs = np.expand_dims(inputs, -1)

        return self._sess.run(self._logits, feed_dict = {self._X: inputs})
