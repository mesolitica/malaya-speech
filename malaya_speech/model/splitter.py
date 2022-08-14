import numpy as np
from malaya_speech.model.frame import Frame
from malaya_speech.utils.padding import (
    sequence_nd as padding_sequence_nd,
)
from malaya_speech.utils.featurization import universal_mel
from malaya_speech.utils.read import resample
from malaya_speech.utils.speechsplit import (
    quantize_f0_numpy,
    get_f0_sptk,
    get_f0_pyworld,
)
from malaya_speech.model.abstract import Abstract


class Split_Wav(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def predict(self, input):
        """
        Split an audio into 4 different speakers.

        Parameters
        ----------
        input: np.array or malaya_speech.model.frame.Frame

        Returns
        -------
        result: np.array
        """
        if isinstance(input, Frame):
            input = input.array

        r = self._execute(
            inputs=[np.expand_dims([input], axis=-1)],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        r = r['logits']
        return r[:, 0, :, 0]

    def __call__(self, input):
        return self.predict(input)


class Split_Mel(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, model, name):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self.__model__ = model
        self.__name__ = name

    def _to_mel(self, y):
        mel = universal_mel(y)
        mel[mel <= np.log(1e-2)] = np.log(1e-2)
        return mel

    def predict(self, input):
        """
        Split an audio into 4 different speakers.

        Parameters
        ----------
        input: np.array or malaya_speech.model.frame.Frame

        Returns
        -------
        result: np.array
        """
        if isinstance(input, Frame):
            input = input.array

        input = self._to_mel(input)

        r = self._execute(
            inputs=[input],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits'],
        )
        r = r['logits']
        return r[:, 0]

    def __call__(self, input):
        return self.predict(input)


class FastSpeechSplit(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        speaker_vector,
        gender_model,
        sess,
        model,
        name,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._speaker_vector = speaker_vector
        self._gender_model = gender_model
        self._sess = sess
        self.__model__ = model
        self.__name__ = name
        self._modes = {'R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU'}
        self._freqs = {'female': [100, 600], 'male': [50, 250]}

    def _get_data(self, x, sr=22050, target_sr=16000):
        x_16k = resample(x, sr, target_sr)
        if self._gender_model is not None:
            gender = self._gender_model(x_16k)
            lo, hi = self._freqs.get(gender, [50, 250])
            f0 = get_f0_sptk(x, lo, hi)
        else:
            f0 = get_f0_pyworld(x)
        f0 = np.expand_dims(f0, -1)
        mel = universal_mel(x)
        v = self._speaker_vector([x_16k])[0]
        v = v / v.max()

        if len(mel) > len(f0):
            mel = mel[: len(f0)]
        return x, mel, f0, v

    def predict(
        self,
        original_audio,
        target_audio,
        modes=['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU'],
    ):
        """
        Change original voice audio to follow targeted voice.

        Parameters
        ----------
        original_audio: np.array or malaya_speech.model.frame.Frame
        target_audio: np.array or malaya_speech.model.frame.Frame
        modes: List[str], optional (default = ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU'])
            R denotes rhythm, F denotes pitch target, U denotes speaker target (vector).

            * ``'R'`` - maintain `original_audio` F and U on `target_audio` R.
            * ``'F'`` - maintain `original_audio` R and U on `target_audio` F.
            * ``'U'`` - maintain `original_audio` R and F on `target_audio` U.
            * ``'RF'`` - maintain `original_audio` U on `target_audio` R and F.
            * ``'RU'`` - maintain `original_audio` F on `target_audio` R and U.
            * ``'FU'`` - maintain `original_audio` R on `target_audio` F and U.
            * ``'RFU'`` - no conversion happened, just do encoder-decoder on `target_audio`

        Returns
        -------
        result: Dict[modes]
        """
        s = set(modes) - self._modes
        if len(s):
            raise ValueError(
                f"{list(s)} not an element of ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU']"
            )

        original_audio = (
            input.array if isinstance(original_audio, Frame) else original_audio
        )
        target_audio = (
            input.array if isinstance(target_audio, Frame) else target_audio
        )
        wav, mel, f0, v = self._get_data(original_audio)
        wav_1, mel_1, f0_1, v_1 = self._get_data(target_audio)
        mels, mel_lens = padding_sequence_nd(
            [mel, mel_1], dim=0, return_len=True
        )
        f0s, f0_lens = padding_sequence_nd(
            [f0, f0_1], dim=0, return_len=True
        )

        f0_org_quantized = quantize_f0_numpy(f0s[0, :, 0])[0]
        f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
        uttr_f0_org = np.concatenate([mels[:1], f0_org_onehot], axis=-1)
        f0_trg_quantized = quantize_f0_numpy(f0s[1, :, 0])[0]
        f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]

        r = self._execute(
            inputs=[mels[:1], f0_trg_onehot, [len(f0s[0])]],
            input_labels=['X', 'f0_onehot', 'len_X'],
            output_labels=['f0_target'],
        )
        f0_pred = r['f0_target']
        f0_pred_quantized = f0_pred.argmax(axis=-1).squeeze(0)
        f0_con_onehot = np.zeros_like(f0_pred)
        f0_con_onehot[0, np.arange(f0_pred.shape[1]), f0_pred_quantized] = 1
        uttr_f0_trg = np.concatenate([mels[:1], f0_con_onehot], axis=-1)
        results = {}
        for condition in modes:
            if condition == 'R':
                uttr_f0_ = uttr_f0_org
                v_ = v
                x_ = mels[1:]
            if condition == 'F':
                uttr_f0_ = uttr_f0_trg
                v_ = v
                x_ = mels[:1]
            if condition == 'U':
                uttr_f0_ = uttr_f0_org
                v_ = v_1
                x_ = mels[:1]
            if condition == 'RF':
                uttr_f0_ = uttr_f0_trg
                v_ = v
                x_ = mels[1:]
            if condition == 'RU':
                uttr_f0_ = uttr_f0_org
                v_ = v_1
                x_ = mels[:1]
            if condition == 'FU':
                uttr_f0_ = uttr_f0_trg
                v_ = v_1
                x_ = mels[:1]
            if condition == 'RFU':
                uttr_f0_ = uttr_f0_trg
                v_ = v_1
                x_ = mels[:1]

            r = self._execute(
                inputs=[uttr_f0_, x_, [v_], [len(f0s[0])]],
                input_labels=['uttr_f0', 'X', 'V', 'len_X'],
                output_labels=['mel_outputs'],
            )
            mel_outputs = r['mel_outputs'][0]
            if 'R' in condition:
                length = mel_lens[1]
            else:
                length = mel_lens[0]
            mel_outputs = mel_outputs[:length]
            results[condition] = mel_outputs

        return results
