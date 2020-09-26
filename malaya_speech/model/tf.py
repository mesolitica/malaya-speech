import tensorflow as tf
import numpy as np
from malaya_speech.model.frame import FRAME
from malaya_speech.utils.padding import padding_sequence_nd


class SPEAKER2VEC:
    def __init__(self, X, logits, vectorizer, sess, model, extra, label, name):
        self._X = X
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._model = model
        self._extra = extra
        self.__name__ = name

    def vectorize(self, inputs):
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        inputs = padding_sequence_nd(inputs)

        if 'vggvox' in self._model:
            inputs = np.expand_dims(inputs, -1)

        return self._sess.run(self._logits, feed_dict = {self._X: inputs})

    def __call__(self, inputs):
        return self.vectorize(inputs)


class CLASSIFICATION:
    def __init__(self, X, logits, vectorizer, sess, model, extra, label, name):
        self._X = X
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._model = model
        self._extra = extra
        self._label = label
        self.__name__ = name

    def predict_proba(self, inputs):
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        inputs = padding_sequence_nd(inputs)

        if 'vggvox' in self._model:
            inputs = np.expand_dims(inputs, -1)

        return self._sess.run(self._logits, feed_dict = {self._X: inputs})

    def predict(self, inputs):
        probs = np.argmax(self.predict_proba(inputs), axis = 1)
        return [self._label[p] for p in probs]

    def __call__(self, input):
        return self.predict([input])[0]
