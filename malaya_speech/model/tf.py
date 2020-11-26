import tensorflow as tf
import numpy as np
from malaya_speech.utils import featurization
from malaya_speech.model.frame import FRAME
from malaya_speech.utils.padding import (
    sequence_nd as padding_sequence_nd,
    sequence_1d,
)
from malaya_speech.utils.char import decode as char_decode


class ABSTRACT:
    def __str__(self):
        return f'<{self.__name__}: {self.__model__}>'


class SPEAKERNET(ABSTRACT):
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


class SPEAKER2VEC(ABSTRACT):
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


class SPEAKERNET_CLASSIFICATION(ABSTRACT):
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


class CLASSIFICATION(ABSTRACT):
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


class UNET(ABSTRACT):
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


class UNET_STFT(ABSTRACT):
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


class STT(ABSTRACT):
    def __init__(
        self, X, X_len, logits, seq_lens, featurizer, vocab, sess, model, name
    ):
        self._X = X
        self._X_len = X_len
        self._logits = logits
        self._seq_lens = seq_lens
        self._featurizer = featurizer
        self._vocab = vocab
        self._softmax = tf.nn.softmax(logits)
        self._sess = sess
        self.__model__ = model
        self.__name__ = name
        self._decoder = None
        self._beam_size = 0

    def _check_decoder(self, decoder, beam_size):
        decoder = decoder.lower()
        if decoder not in ['greedy', 'beam']:
            raise ValueError('mode only supports [`greedy`, `beam`]')
        if beam_size < 1:
            raise ValueError('beam_size must bigger than 0')
        return decoder

    def predict(
        self, inputs, decoder: str = 'beam', beam_size: int = 100, **kwargs
    ):
        """
        Transcribe inputs, will return list of strings.

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
        decoder = self._check_decoder(decoder, beam_size)
        padded, lens = sequence_1d(inputs, return_len = True)

        if decoder == 'greedy':
            beam_size = 1
        if beam_size != self._beam_size:
            self._beam_size = beam_size
            self._decoded = tf.nn.ctc_beam_search_decoder(
                self._logits,
                self._seq_lens,
                beam_width = self._beam_size,
                top_paths = 1,
                merge_repeated = True,
                **kwargs,
            )[0][0]

        r = self._sess.run(
            self._decoded, feed_dict = {self._X: padded, self._X_len: lens}
        )
        decoded = np.zeros(r.dense_shape, dtype = np.int32)
        for i in range(r.values.shape[0]):
            decoded[r.indices[i][0], r.indices[i][1]] = r.values[i]

        results = []
        for i in range(len(decoded)):
            results.append(
                char_decode(decoded[i], lookup = self._vocab).replace(
                    '<PAD>', ''
                )
            )
        return results

    def predict_lm(self, inputs, lm, beam_size: int = 100, **kwargs):
        """
        Transcribe inputs using Beam Search + LM, will return list of strings.
        This method will not able to utilise batch decoding, instead will do loop to decode for each elements.

        Parameters
        ----------
        input: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.FRAME].
        lm: ctc_decoders.Scorer
            Returned from `malaya_speech.stt.language_model()`.
        beam_size: int, optional (default=100)
            beam size for beam decoder.
        

        Returns
        -------
        result: List[str]
        """
        try:
            from ctc_decoders import ctc_beam_search_decoder
        except:
            raise ModuleNotFoundError(
                'ctc_decoders not installed. Please install it by `pip install ctc-decoders` and try again.'
            )
        padded, lens = sequence_1d(inputs, return_len = True)
        logits, seq_lens = self._sess.run(
            [self._softmax, self._seq_lens],
            feed_dict = {self._X: padded, self._X_len: lens},
        )
        logits = np.transpose(logits, axes = (1, 0, 2))
        results = []
        for i in range(len(logits)):
            d = ctc_beam_search_decoder(
                logits[i][: seq_lens[i]],
                self._vocab,
                beam_size,
                ext_scoring_func = lm,
                **kwargs,
            )
            results.append(d[0][1])
        return results

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
