import tensorflow as tf
from malaya_speech.model.frame import FRAME


class SPEAKER2VEC:
    def __init__(self, X, vectorizer, sess):
        self._X = X
        self._vectorizer = vectorizer
        self._sess = sess

    def predict(self, inputs):
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]
