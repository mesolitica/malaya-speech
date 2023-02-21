import torch
import yaml
import numpy as np
from malaya_speech.utils.padding import sequence_1d
from malaya_speech.model.frame import Frame
from malaya_speech.utils import nemo_featurization
from malaya_speech.utils.nemo_featurization import (
    AudioToMelSpectrogramPreprocessor,
)
from malaya_speech.nemo import conv_asr
from malaya_speech.nemo.conv_asr import SpeakerDecoder
from malaya_speech.utils.activation import softmax
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy


class SpeakerVector(torch.nn.Module):
    def __init__(self, config, pth, model, name):
        super().__init__()

        with open(config) as stream:
            try:
                d = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError('invalid yaml')

        preprocessor = d['preprocessor'].copy()
        preprocessor.pop('_target_')

        encoder = d['encoder'].copy()
        encoder_target = encoder.pop('_target_').split('.')[-1]

        decoder = d['decoder'].copy()
        decoder.pop('_target_')

        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preprocessor)
        self.encoder = getattr(conv_asr, encoder_target)(**encoder)
        self.decoder = SpeakerDecoder(**decoder)

        self.load_state_dict(torch.load(pth, map_location='cpu'))

        self.__model__ = model
        self.__name__ = name

    def forward(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        cuda = next(self.parameters()).is_cuda
        inputs, lengths = sequence_1d(
            inputs, return_len=True
        )
        inputs = to_tensor_cuda(torch.Tensor(inputs.astype(np.float32)), cuda)
        lengths = to_tensor_cuda(torch.Tensor(lengths), cuda)
        o_processor = self.preprocessor(inputs, lengths)
        o_encoder = self.encoder(*o_processor)
        return self.decoder(*o_encoder)

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
        """
        r = self.forward(inputs=inputs)
        return to_numpy(r[1])

    def __call__(self, inputs):
        return self.vectorize(inputs)


class Classification(torch.nn.Module):
    def __init__(self, config, pth, label, model, name):
        super().__init__()

        with open(config) as stream:
            try:
                d = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError('invalid yaml')

        preprocessor = d['preprocessor'].copy()
        preprocessor_target = (preprocessor.pop('_target_', None)
                               or preprocessor.pop('cls', None)).split('.')[-1]
        if 'params' in preprocessor:
            preprocessor = preprocessor['params']

        encoder = d['encoder'].copy()
        encoder_target = (encoder.pop('_target_', None) or encoder.pop('cls', None)).split('.')[-1]
        if 'params' in encoder:
            encoder = encoder['params']

        decoder = d['decoder'].copy()
        decoder_target = (decoder.pop('_target_', None) or decoder.pop('cls', None)).split('.')[-1]
        if 'params' in decoder:
            decoder = decoder['params']

        self.preprocessor = getattr(nemo_featurization, preprocessor_target)(**preprocessor)
        self.encoder = getattr(conv_asr, encoder_target)(**encoder)
        self.decoder = getattr(conv_asr, decoder_target)(**decoder)

        self.load_state_dict(torch.load(pth, map_location='cpu'))

        self.labels = label
        self.__model__ = model
        self.__name__ = name

    def forward(self, inputs):
        """
        Vectorize inputs.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        """
        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]
        cuda = next(self.parameters()).is_cuda
        inputs, lengths = sequence_1d(
            inputs, return_len=True
        )
        inputs = to_tensor_cuda(torch.Tensor(inputs.astype(np.float32)), cuda)
        lengths = to_tensor_cuda(torch.Tensor(lengths), cuda)
        o_processor = self.preprocessor(inputs, lengths)
        o_encoder = self.encoder(*o_processor)
        try:
            r = self.decoder(*o_encoder)
        except BaseException:
            r = self.decoder(o_encoder[0])
        return r

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
        o = self.forward(inputs=inputs)
        if isinstance(o, tuple):
            o = o[0]
        r = to_numpy(o)
        return softmax(r, axis=-1)

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
        o = self.forward(inputs=inputs)
        if isinstance(o, tuple):
            o = o[0]
        r = to_numpy(o)
        probs = np.argmax(r, axis=1)
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
