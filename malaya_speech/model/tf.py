import tensorflow as tf
import numpy as np
from malaya_speech.utils import featurization
from malaya_speech.model.frame import FRAME
from malaya_speech.utils.padding import padding_sequence_nd


class SPEAKER2VEC:
    def __init__(self, X, logits, vectorizer, sess, model, extra, label, name):
        self._X = X
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.__name__ = name
        self.__model__ = model

    def vectorize(self, inputs):
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        inputs = padding_sequence_nd(inputs)

        if 'vggvox' in self.__model__:
            inputs = np.expand_dims(inputs, -1)

        return self._sess.run(self._logits, feed_dict = {self._X: inputs})

    def __call__(self, inputs):
        return self.vectorize(inputs)


class CLASSIFICATION:
    def __init__(self, X, logits, vectorizer, sess, model, extra, label, name):
        self._X = X
        self._logits = tf.nn.softmax(logits)
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self._label = label
        self.__model__ = model
        self.__name__ = name

    def predict_proba(self, inputs):
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        inputs = padding_sequence_nd(inputs)

        if 'vggvox' in self.__model__:
            inputs = np.expand_dims(inputs, -1)

        return self._sess.run(self._logits, feed_dict = {self._X: inputs})

    def predict(self, inputs):
        probs = np.argmax(self.predict_proba(inputs), axis = 1)
        return [self._label[p] for p in probs]

    def __call__(self, input):
        return self.predict([input])[0]


class UNET:
    def __init__(self, X, logits, sess, model, name):
        self._X = X
        self._logits = logits
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, inputs):
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]
        mels = [featurization.scale_mel(s).T for s in inputs]
        x, lens = padding_sequence_nd(
            mels, maxlen = 256, dim = 0, return_len = True
        )
        l = self._sess.run(self._logits, feed_dict = {self._X: x})
        results = []
        for index in range(len(x)):
            results.append(
                featurization.unscale_mel(
                    x[index, : lens[index]].T + l[index, : lens[index], :, 0].T
                )
            )
        return results

    def __call__(self, inputs):
        return self.predict(inputs)
