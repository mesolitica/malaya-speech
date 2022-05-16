import numpy as np
from malaya_speech.model.frame import Frame
from malaya_speech.utils.padding import (
    sequence_nd as padding_sequence_nd,
    sequence_1d,
)
from malaya_speech.utils.activation import softmax
from malaya_speech.model.abstract import Abstract


class Transformer2Vec(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        model,
        label,
        name,
        vectorizer=None,
        extra=None,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """

        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        inputs, lengths = padding_sequence_nd(
            inputs, dim=0, return_len=True
        )

        r = self._execute(
            inputs=[inputs, lengths],
            input_labels=['X_placeholder', 'X_len_placeholder'],
            output_labels=['logits'],
        )
        return r['logits']

    def __call__(self, inputs):
        return self.vectorize(inputs)


class Speakernet(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input) for input in inputs]
        inputs, lengths = padding_sequence_nd(
            inputs, dim=0, return_len=True
        )

        r = self._execute(
            inputs=[inputs, lengths],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits'],
        )
        return r['logits']

    def __call__(self, inputs):
        return self.vectorize(inputs)


class Speaker2Vec(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]

        if self.__model__ == 'deep-speaker':
            dim = 0
        else:
            dim = 1
        inputs = padding_sequence_nd(inputs, dim=dim)
        inputs = np.expand_dims(inputs, -1)

        r = self._execute(
            inputs=[inputs],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return r['logits']

    def __call__(self, inputs):
        return self.vectorize(inputs)


class SpeakernetClassification(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def predict_proba(self, inputs):
        """
        Predict inputs, will return probability.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input) for input in inputs]
        inputs, lengths = padding_sequence_nd(
            inputs, dim=0, return_len=True
        )

        r = self._execute(
            inputs=[inputs, lengths],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    def predict(self, inputs):
        """
        Predict inputs, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: List[str]
            returned [B].
        """
        probs = np.argmax(self.predict_proba(inputs), axis=1)
        return [self.labels[p] for p in probs]

    def __call__(self, input):
        """
        Predict input, will return label.

        Parameters
        ----------
        inputs: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: str
        """

        return self.predict([input])[0]


class Classification(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def predict_proba(self, inputs):
        """
        Predict inputs, will return probability.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        if self.__model__ == 'deep-speaker':
            dim = 0
        else:
            dim = 1

        inputs = padding_sequence_nd(inputs, dim=dim)
        inputs = np.expand_dims(inputs, -1)

        r = self._execute(
            inputs=[inputs],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    def predict(self, inputs):
        """
        Predict inputs, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].

        Returns
        -------
        result: List[str]
            returned [B].
        """
        probs = np.argmax(self.predict_proba(inputs), axis=1)
        return [self.labels[p] for p in probs]

    def __call__(self, input):
        """
        Predict input, will return label.

        Parameters
        ----------
        inputs: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: str
        """
        return self.predict([input])[0]


class MarbleNetClassification(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        vectorizer,
        sess,
        model,
        extra,
        label,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def predict_proba(self, inputs):
        """
        Predict inputs, will return probability.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        inputs, lengths = sequence_1d(
            inputs, return_len=True
        )

        r = self._execute(
            inputs=[inputs, lengths],
            input_labels=['X_placeholder', 'X_len_placeholder'],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    def predict(self, inputs):
        """
        Predict inputs, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        Returns
        -------
        result: List[str]
            returned [B].
        """
        probs = np.argmax(self.predict_proba(inputs), axis=1)
        return [self.labels[p] for p in probs]

    def __call__(self, input):
        """
        Predict input, will return label.

        Parameters
        ----------
        inputs: np.array
            np.array or malaya_speech.model.frame.Frame.
        Returns
        -------
        result: str
        """

        return self.predict([input])[0]
