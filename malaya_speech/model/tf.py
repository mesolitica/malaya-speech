import tensorflow as tf
import numpy as np
from malaya_speech.utils import featurization
from malaya_speech.model.frame import Frame
from malaya_speech.utils.padding import (
    sequence_nd as padding_sequence_nd,
    sequence_1d,
)
from malaya_speech.utils.char import decode as char_decode
from malaya_speech.utils.subword import decode as subword_decode
from malaya_speech.utils.beam_search import transducer as transducer_beam
from malaya_speech.utils.execute import execute_graph
from malaya_speech.utils.activation import softmax
from malaya_speech.utils.constant import MEL_MEAN, MEL_STD


class Abstract:
    def __str__(self):
        return f'<{self.__name__}: {self.__model__}>'

    def _execute(self, inputs, input_labels, output_labels):
        return execute_graph(
            inputs = inputs,
            input_labels = input_labels,
            output_labels = output_labels,
            sess = self._sess,
            input_nodes = self._input_nodes,
            output_nodes = self._output_nodes,
        )


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
            inputs, dim = 0, return_len = True
        )

        r = self._execute(
            inputs = [inputs, lengths],
            input_labels = ['Placeholder', 'Placeholder_1'],
            output_labels = ['logits'],
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
        inputs = padding_sequence_nd(inputs, dim = dim)
        inputs = np.expand_dims(inputs, -1)

        r = self._execute(
            inputs = [inputs],
            input_labels = ['Placeholder'],
            output_labels = ['logits'],
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
            inputs, dim = 0, return_len = True
        )

        r = self._execute(
            inputs = [inputs, lengths],
            input_labels = ['Placeholder', 'Placeholder_1'],
            output_labels = ['logits'],
        )
        return softmax(r['logits'], axis = -1)

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
        probs = np.argmax(self.predict_proba(inputs), axis = 1)
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

        inputs = padding_sequence_nd(inputs, dim = dim)
        inputs = np.expand_dims(inputs, -1)

        r = self._execute(
            inputs = [inputs],
            input_labels = ['Placeholder'],
            output_labels = ['logits'],
        )
        return softmax(r['logits'], axis = -1)

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
        probs = np.argmax(self.predict_proba(inputs), axis = 1)
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


class UNET(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
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
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        mels = [featurization.scale_mel(s).T for s in inputs]
        x, lens = padding_sequence_nd(
            mels, maxlen = 256, dim = 0, return_len = True
        )

        r = self._execute(
            inputs = [x],
            input_labels = ['Placeholder'],
            output_labels = ['logits'],
        )
        l = r['logits']

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


class UNETSTFT(Abstract):
    def __init__(
        self, input_nodes, output_nodes, instruments, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
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
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: Dict
        """
        if isinstance(input, Frame):
            input = input.array

        r = self._execute(
            inputs = [input],
            input_labels = ['Placeholder'],
            output_labels = list(self._output_nodes.keys()),
        )
        results = {}
        for no, instrument in enumerate(self._instruments):
            results[instrument] = r[f'logits_{no}']
        return results

    def __call__(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: Dict
        """
        return self.predict(input)


class UNET1D(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: np.array
        """
        if isinstance(input, Frame):
            input = input.array

        r = self._execute(
            inputs = [input],
            input_labels = ['Placeholder'],
            output_labels = ['logits'],
        )
        return r['logits']

    def __call__(self, input):
        """
        Enhance inputs, will return waveform.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: np.array
        """
        return self.predict(input)


class Transducer(Abstract):
    def __init__(
        self,
        X_placeholder,
        X_len_placeholder,
        encoded_placeholder,
        predicted_placeholder,
        states_placeholder,
        padded_features,
        padded_lens,
        encoded,
        ytu,
        new_states,
        initial_states,
        greedy,
        featurizer,
        vocab,
        time_reduction_factor,
        sess,
        model,
        name,
    ):
        self._X_placeholder = X_placeholder
        self._X_len_placeholder = X_len_placeholder
        self._encoded_placeholder = encoded_placeholder
        self._predicted_placeholder = predicted_placeholder
        self._states_placeholder = states_placeholder

        self._padded_features = padded_features
        self._padded_lens = padded_lens
        self._encoded = encoded
        self._ytu = ytu
        self._new_states = new_states
        self._initial_states = initial_states

        self._greedy = greedy
        self._featurizer = featurizer
        self._vocab = vocab
        self._time_reduction_factor = time_reduction_factor
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def _check_decoder(self, decoder, beam_size):
        decoder = decoder.lower()
        if decoder not in ['greedy', 'beam']:
            raise ValueError('mode only supports [`greedy`, `beam`]')
        if beam_size < 1:
            raise ValueError('beam_size must bigger than 0')
        return decoder

    def predict_timestamp(self, input):
        """
        Transcribe input and get timestamp, only support greedy decoder.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.

        Returns
        -------
        result: Dict[timestamp, decoded]
        """
        input = input.array if isinstance(input, Frame) else input

        padded, lens = sequence_1d([inputs], return_len = True)

    def predict(
        self, inputs, decoder: str = 'greedy', beam_size: int = 5, **kwargs
    ):
        """
        Transcribe inputs, will return list of strings.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        decoder: str, optional (default='beam')
            decoder mode, allowed values:

            * ``'greedy'`` - greedy decoder.
            * ``'beam'`` - beam decoder.
        beam_size: int, optional (default=5)
            beam size for beam decoder.

        Returns
        -------
        result: List[str]
        """
        decoder = self._check_decoder(decoder, beam_size)

        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        padded, lens = sequence_1d(inputs, return_len = True)
        results = []

        if decoder == 'greedy':
            r = self._sess.run(
                self._greedy,
                feed_dict = {
                    self._X_placeholder: padded,
                    self._X_len_placeholder: lens,
                },
            )
            for row in r:
                results.append(subword_decode(self._vocab, row[row > 0]))

        else:
            encoded_, padded_lens_, s = self._sess.run(
                [self._encoded, self._padded_lens, self._initial_states],
                feed_dict = {
                    self._X_placeholder: padded,
                    self._X_len_placeholder: lens,
                },
            )
            padded_lens_ = padded_lens_ // self._time_reduction_factor
            for i in range(len(encoded_)):
                r = transducer_beam(
                    enc = encoded_[i],
                    total = padded_lens_[i],
                    initial_states = s,
                    encoded_placeholder = self._encoded_placeholder,
                    predicted_placeholder = self._predicted_placeholder,
                    states_placeholder = self._states_placeholder,
                    ytu = self._ytu,
                    new_states = self._new_states,
                    sess = self._sess,
                    beam_width = beam_size,
                    **kwargs,
                )
                results.append(subword_decode(self._vocab, r))
        return results

    def __call__(self, input, decoder: str = 'greedy', **kwargs):
        """
        Transcribe input, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        decoder: str, optional (default='beam')
            decoder mode, allowed values:

            * ``'greedy'`` - greedy decoder.
            * ``'beam'`` - beam decoder.
        **kwargs: keyword arguments passed to `predict`.

        Returns
        -------
        result: str
        """
        return self.predict([input], decoder = decoder, **kwargs)[0]


class Vocoder(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, inputs):
        """
        Change Mel to Waveform.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        Returns
        -------
        result: List
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        padded, lens = sequence_1d(inputs, return_len = True)

        r = self._execute(
            inputs = [padded],
            input_labels = ['Placeholder'],
            output_labels = ['logits'],
        )
        return r['logits'][:, :, 0]

    def __call__(self, input):
        return self.predict([input])[0]


class Tacotron(Abstract):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._stats = stats
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, string, **kwargs):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str

        Returns
        -------
        result: Dict[string, decoder-output, postnet-output, universal-output, alignment]
        """

        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs = [[ids], [len(ids)]],
            input_labels = ['Placeholder', 'Placeholder_1'],
            output_labels = [
                'decoder_output',
                'post_mel_outputs',
                'alignment_histories',
            ],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'postnet-output': r['post_mel_outputs'][0],
            'universal-output': v,
            'alignment': r['alignment_histories'][0],
        }

    def __call__(self, input):
        return self.predict(input)


class Fastspeech(Abstract):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._stats = stats
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(
        self,
        string,
        speed_ratio: float = 1.0,
        f0_ratio: float = 1.0,
        energy_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        speed_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.
        f0_ratio: float, optional (default=1.0)
            Increase this variable will increase frequency, low frequency will generate more deeper voice.
        energy_ratio: float, optional (default=1.0)
            Increase this variable will increase loudness.

        Returns
        -------
        result: Dict[string, decoder-output, postnet-output, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs = [[ids], [speed_ratio], [f0_ratio], [energy_ratio]],
            input_labels = [
                'Placeholder',
                'speed_ratios',
                'f0_ratios',
                'energy_ratios',
            ],
            output_labels = ['decoder_output', 'post_mel_outputs'],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'postnet-output': r['post_mel_outputs'][0],
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class FastVC(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        waveform_to_mel,
        speaker_vector,
        sess,
        model,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._waveform_to_mel = waveform_to_mel
        self._speaker_vector = speaker_vector
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, original_audio, target_audio):
        """
        Change original voice audio to follow targeted voice.

        Parameters
        ----------
        original_audio: np.array or malaya_speech.model.frame.Frame
        target_audio: np.array or malaya_speech.model.frame.Frame

        Returns
        -------
        result: Dict[decoder-output, postnet-output]
        """
        original_audio = (
            input.array if isinstance(original_audio, Frame) else original_audio
        )
        target_audio = (
            input.array if isinstance(target_audio, Frame) else target_audio
        )

        original_mel = self._waveform_to_mel(original_audio)
        target_mel = self._waveform_to_mel(target_audio)

        original_v = self._speaker_vector([original_audio])[0] * 30 - 3.5
        target_v = self._speaker_vector([target_audio])[0] * 30 - 3.5

        r = self._execute(
            inputs = [
                [original_mel],
                [original_v],
                [target_v],
                [len(original_mel)],
            ],
            input_labels = [
                'mel',
                'ori_vector',
                'target_vector',
                'mel_lengths',
            ],
            output_labels = ['mel_before', 'mel_after'],
        )
        return {
            'decoder-output': r['mel_before'][0],
            'postnet-output': r['mel_after'][0],
        }

    def __call__(self, original_audio, target_audio):
        return self.predict(original_audio, target_audio)
