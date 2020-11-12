import tensorflow as tf
import numpy as np
from malaya_speech.utils import featurization
from malaya_speech.model.frame import FRAME
from malaya_speech.utils.padding import sequence_nd as padding_sequence_nd


class SPEAKERNET:
    def __init__(
        self, X, X_len, logits, vectorizer, sess, model, extra, label, name
    ):
        self._X = X
        self._X_len = X_len
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.__model__ = model
        self.__name__ = name

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.FRAME].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input) for input in inputs]
        inputs, lengths = padding_sequence_nd(
            inputs, dim = 0, return_len = True
        )

        return self._sess.run(
            self._logits, feed_dict = {self._X: inputs, self._X_len: lengths}
        )

    def __call__(self, inputs):
        return self.vectorize(inputs)


class SPEAKER2VEC:
    def __init__(self, X, logits, vectorizer, sess, model, extra, label, name):
        self._X = X
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._extra = extra
        self.__model__ = model
        self.__name__ = name

    def vectorize(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.FRAME].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]

        if self.__model__ == 'deep-speaker':
            dim = 0
        else:
            dim = 1
        inputs = padding_sequence_nd(inputs, dim = dim)
        inputs = np.expand_dims(inputs, -1)

        return self._sess.run(self._logits, feed_dict = {self._X: inputs})

    def __call__(self, inputs):
        return self.vectorize(inputs)


class SPEAKERNET_CLASSIFICATION:
    def __init__(
        self, X, X_len, logits, vectorizer, sess, model, extra, label, name
    ):
        self._X = X
        self._X_len = X_len
        self._logits = tf.nn.softmax(logits)
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
            List[np.array] or List[malaya_speech.model.frame.FRAME].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input) for input in inputs]
        inputs, lengths = padding_sequence_nd(
            inputs, dim = 0, return_len = True
        )

        return self._sess.run(
            self._logits, feed_dict = {self._X: inputs, self._X_len: lengths}
        )

    def predict(self, inputs):
        """
        Predict inputs, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.FRAME].

        Returns
        -------
        result: List[str]
            returned [B].
        """
        probs = np.argmax(self.predict_proba(inputs), axis = 1)
        return [self.labels[p] for p in probs]

    def __call__(self, input):
        """
        Predict input, will return label.

        Parameters
        ----------
        inputs: np.array
            np.array or malaya_speech.model.frame.FRAME.

        Returns
        -------
        result: str
        """

        return self.predict([input])[0]


class CLASSIFICATION:
    def __init__(self, X, logits, vectorizer, sess, model, extra, label, name):
        self._X = X
        self._logits = tf.nn.softmax(logits)
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
            List[np.array] or List[malaya_speech.model.frame.FRAME].

        Returns
        -------
        result: np.array
            returned [B, D].
        """
        inputs = [
            input.array if isinstance(input, FRAME) else input
            for input in inputs
        ]

        inputs = [self._vectorizer(input, **self._extra) for input in inputs]
        if self.__model__ == 'deep-speaker':
            dim = 0
        else:
            dim = 1
        inputs = padding_sequence_nd(inputs, dim = dim)
        inputs = np.expand_dims(inputs, -1)

        return self._sess.run(self._logits, feed_dict = {self._X: inputs})

    def predict(self, inputs):
        """
        Predict inputs, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.FRAME].

        Returns
        -------
        result: List[str]
            returned [B].
        """
        probs = np.argmax(self.predict_proba(inputs), axis = 1)
        return [self.labels[p] for p in probs]

    def __call__(self, input):
        """
        Predict input, will return label.

        Parameters
        ----------
        inputs: np.array
            np.array or malaya_speech.model.frame.FRAME.

        Returns
        -------
        result: str
        """
        return self.predict([input])[0]


class UNET:
    def __init__(self, X, logits, sess, model, name):
        self._X = X
        self._logits = logits
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, inputs):
        """
        Enhance inputs, will return melspectrogram.

        Parameters
        ----------
        inputs: List[np.array]

        Returns
        -------
        result: List
        """
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


class UNET_STFT:
    def __init__(self, X, logits, instruments, sess, model, name):
        self._X = X
        self._logits = {
            instrument: logits[no] for no, instrument in enumerate(instruments)
        }
        self._instruments = instruments
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.FRAME.

        Returns
        -------
        result: Dict
        """
        if isinstance(input, FRAME):
            input = input.array

        return self._sess.run(self._logits, feed_dict = {self._X: input})

    def __call__(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.FRAME.

        Returns
        -------
        result: Dict
        """
        return self.predict(input)


class STT:
    def __init__(self, X, logits, greedy, beam, sess, model, name):
        self._X = X
        self._logits = logits
        self._greedy = greedy
        self._beam = beam
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, inputs, decoder: str = 'beam'):
        """
        Transcribe inputs, will return list of strings.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.FRAME].
        decoder: str, optional (default='beam')
            decoder mode, allowed values:

            * ``'greedy'`` - greedy decoder.
            * ``'beam'`` - beam decoder with 100 beam width.

        Returns
        -------
        result: List[str]
        """

    def predict_lm(
        self,
        inputs,
        decoder: str = 'beam',
        beam_size: int = 100,
        vocabulary = None,
    ):
        """
        Transcribe inputs using Beam Search + LM, will return list of strings.
        This method will not able to utilise batch decoding, instead will do loop to decode for each elements.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.FRAME].
        decoder: str, optional (default='beam')
            decoder mode, allowed values:

            * ``'greedy'`` - greedy decoder.
            * ``'beam'`` - beam decoder.
        beam_size: int, optional (default=100)
            beam size for beam decoder.
        

        Returns
        -------
        result: List[str]
        """
        try:
            from ctc_decoders import ctc_greedy_decoder, ctc_beam_search_decoder
        except:
            raise ModuleNotFoundError(
                'ctc_decoders not installed. Please install it by compile from https://github.com/usimarit/ctc_decoders and try again.'
            )

    def __call__(
        self, input, decoder: str = 'greedy', lm: bool = False, **kwargs
    ):
        """
        Transcribe input, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.FRAME.
        decoder: str, optional (default='beam')
            decoder mode, allowed values:

            * ``'greedy'`` - greedy decoder.
            * ``'beam'`` - beam decoder.
        lm: bool, optional (default=False)
        **kwargs: keyword arguments passed to `predict` or `predict_lm`.

        Returns
        -------
        result: str
        """
        if lm:
            method = self.predict_lm
        else:
            method = self.predict
        return method([input], decoder = decoder, **kwargs)[0]
