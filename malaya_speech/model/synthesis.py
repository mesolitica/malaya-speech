from malaya_speech.model.frame import Frame
from malaya_speech.utils.padding import (
    sequence_1d,
)
from malaya_speech.utils.astype import float_to_int
from malaya_speech.utils.featurization import universal_mel
from malaya_speech.model.abstract import Abstract, TTS
from malaya_speech.utils.constant import MEL_MEAN, MEL_STD
from typing import Callable


class Vocoder(Abstract, TTS):
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
        padded, lens = sequence_1d(inputs, return_len=True)

        r = self._execute(
            inputs=[padded],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return r['logits'][:, :, 0]

    def __call__(self, input):
        return self.predict([input])[0]


class Tacotron(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        TTS.__init__(self)

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
        result: Dict[string, decoder-output, mel-output, universal-output, alignment]
        """

        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [len(ids)]],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=[
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
            'mel-output': r['post_mel_outputs'][0],
            'universal-output': v,
            'alignment': r['alignment_histories'][0],
        }

    def __call__(self, input):
        return self.predict(input)


class Fastspeech(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        TTS.__init__(self)

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
        result: Dict[string, decoder-output, mel-output, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [speed_ratio], [f0_ratio], [energy_ratio]],
            input_labels=[
                'Placeholder',
                'speed_ratios',
                'f0_ratios',
                'energy_ratios',
            ],
            output_labels=['decoder_output', 'post_mel_outputs'],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'mel-output': r['post_mel_outputs'][0],
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class FastspeechSDP(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        TTS.__init__(self)

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
        temperature_durator: float = 0.6666,
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
        temperature_durator: float, optional (default=0.66666)
            Durator trying to predict alignment with random.normal() * temperature_durator.

        Returns
        -------
        result: Dict[string, decoder-output, mel-output, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], speed_ratio, [f0_ratio], [energy_ratio], temperature_durator],
            input_labels=[
                'Placeholder',
                'speed_ratios',
                'f0_ratios',
                'energy_ratios',
                'noise_scale_w',
            ],
            output_labels=['decoder_output', 'post_mel_outputs'],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'mel-output': r['post_mel_outputs'][0],
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class E2E_FastSpeech(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        TTS.__init__(self, e2e=True)

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
        temperature_durator: float = 0.6666,
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
        temperature_durator: float, optional (default=0.66666)
            Durator trying to predict alignment with random.normal() * temperature_durator.

        Returns
        -------
        result: Dict[string, decoder-output, y]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], speed_ratio, [f0_ratio], [energy_ratio], temperature_durator],
            input_labels=[
                'Placeholder',
                'speed_ratios',
                'f0_ratios',
                'energy_ratios',
                'noise_scale_w',
            ],
            output_labels=['y_hat'],
        )
        return {
            'string': t,
            'ids': ids,
            'y': r['y_hat'],
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class FastVC(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        speaker_vector,
        magnitude,
        sess,
        model,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._speaker_vector = speaker_vector
        self._magnitude = magnitude
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
        result: Dict[decoder-output, mel-output]
        """
        original_audio = (
            input.array if isinstance(original_audio, Frame) else original_audio
        )
        target_audio = (
            input.array if isinstance(target_audio, Frame) else target_audio
        )

        original_mel = universal_mel(original_audio)
        target_mel = universal_mel(target_audio)

        original_v = self._magnitude(self._speaker_vector([original_audio])[0])
        target_v = self._magnitude(self._speaker_vector([target_audio])[0])

        r = self._execute(
            inputs=[
                [original_mel],
                [original_v],
                [target_v],
                [len(original_mel)],
            ],
            input_labels=[
                'mel',
                'ori_vector',
                'target_vector',
                'mel_lengths',
            ],
            output_labels=['mel_before', 'mel_after'],
        )
        return {
            'decoder-output': r['mel_before'][0],
            'mel-output': r['mel_after'][0],
        }

    def __call__(self, original_audio, target_audio):
        return self.predict(original_audio, target_audio)


class Fastpitch(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name
    ):
        TTS.__init__(self)
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
        pitch_ratio: float = 1.0,
        pitch_addition: float = 0.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        speed_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.
        pitch_ratio: float, optional (default=1.0)
            pitch = pitch * pitch_ratio, amplify existing pitch contour.
        pitch_addition: float, optional (default=0.0)
            pitch = pitch + pitch_addition, change pitch contour.

        Returns
        -------
        result: Dict[string, decoder-output, mel-output, pitch-output, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [speed_ratio], [pitch_ratio], [pitch_addition]],
            input_labels=[
                'Placeholder',
                'speed_ratios',
                'pitch_ratios',
                'pitch_addition',
            ],
            output_labels=['decoder_output', 'post_mel_outputs', 'pitch_outputs'],
        )
        v = r['post_mel_outputs'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'decoder-output': r['decoder_output'][0],
            'mel-output': r['post_mel_outputs'][0],
            'pitch-output': r['pitch_outputs'][0],
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class GlowTTS(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, stats, sess, model, name, **kwargs
    ):
        TTS.__init__(self)
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
        temperature: float = 0.3333,
        length_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        temperature: float, optional (default=0.3333)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, mel-output, alignment, universal-output]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        r = self._execute(
            inputs=[[ids], [len(ids)], [temperature], [length_ratio]],
            input_labels=[
                'input_ids',
                'lens',
                'temperature',
                'length_ratio',
            ],
            output_labels=['mel_output', 'alignment_histories'],
        )
        v = r['mel_output'][0] * self._stats[1] + self._stats[0]
        v = (v - MEL_MEAN) / MEL_STD
        return {
            'string': t,
            'ids': ids,
            'mel-output': r['mel_output'][0],
            'alignment': r['alignment_histories'][0].T,
            'universal-output': v,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class GlowTTS_MultiSpeaker(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, speaker_vector, stats, sess, model, name
    ):
        TTS.__init__(self)
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._speaker_vector = speaker_vector
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def _predict(self,
                 string, left_audio, right_audio,
                 temperature: float = 0.3333,
                 length_ratio: float = 1.0,
                 **kwargs):
        t, ids = self._normalizer.normalize(string, **kwargs)
        left_v = self._speaker_vector([left_audio])
        right_v = self._speaker_vector([right_audio])
        r = self._execute(
            inputs=[[ids], [len(ids)], [temperature], [length_ratio], left_v, right_v],
            input_labels=[
                'input_ids',
                'lens',
                'temperature',
                'length_ratio',
                'speakers',
                'speakers_right',
            ],
            output_labels=['mel_output', 'alignment_histories'],
        )
        return {
            'string': t,
            'ids': ids,
            'alignment': r['alignment_histories'][0].T,
            'universal-output': r['mel_output'][0][:-8],
        }

    def predict(
        self,
        string,
        audio,
        temperature: float = 0.3333,
        length_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        audio: np.array
            np.array or malaya_speech.model.frame.Frame, must in 16k format.
            We only trained on `female`, `male`, `husein` and `haqkiem` speakers.
        temperature: float, optional (default=0.3333)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, alignment, universal-output]
        """
        return self._predict(string=string,
                             left_audio=audio, right_audio=audio,
                             temperature=temperature, length_ratio=length_ratio, **kwargs)

    def voice_conversion(self, string, original_audio, target_audio,
                         temperature: float = 0.3333,
                         length_ratio: float = 1.0,
                         **kwargs,):
        """
        Change string to Mel.

        Parameters
        ----------
        string: str
        original_audio: np.array
            original speaker to encode speaking style, must in 16k format.
        target_audio: np.array
            target speaker to follow speaking style from `original_audio`, must in 16k format.
        temperature: float, optional (default=0.3333)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, alignment, universal-output]
        """
        return self._predict(string=string,
                             left_audio=original_audio, right_audio=target_audio,
                             temperature=temperature, length_ratio=length_ratio, **kwargs)

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)


class VITS(Abstract, TTS):
    def __init__(
        self, input_nodes, output_nodes, normalizer, sess, model, name
    ):
        TTS.__init__(self, e2e=True)

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._normalizer = normalizer
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(
        self,
        string,
        temperature: float = 0.5,
        temperature_durator: float = 0.5,
        length_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to waveform.

        Parameters
        ----------
        string: str
        temperature: float, optional (default=0.5)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        temperature_durator: float, optional (default=1.0)
            Durator trying to predict alignment with random.normal() * temperature_durator.
            Only useful for SDP-based models.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, mel-output, alignment, y]
        """
        t, ids = self._normalizer.normalize(string, **kwargs)
        inputs = [[ids], [len(ids)], [temperature], [length_ratio]]
        input_labels = [
            'input_ids',
            'lens',
            'temperature',
            'length_ratio',
        ]
        if 'sdp' in self.__model__:
            inputs.append([temperature_durator])
            input_labels.append('noise_scale_w')
        r = self._execute(
            inputs=inputs,
            input_labels=input_labels,
            output_labels=['mel_output', 'alignment_histories', 'y_hat'],
        )
        return {
            'string': t,
            'ids': ids,
            'mel-output': r['mel_output'],
            'alignment': r['alignment_histories'][0].T,
            'y': r['y_hat'],
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)
