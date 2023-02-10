import torch
import yaml
import numpy as np
from malaya_speech.utils.padding import sequence_1d
from malaya_speech.model.frame import Frame
from malaya_speech.utils.nemo_featurization import AudioToMelSpectrogramPreprocessor
from malaya_speech.nemo.conv_asr import ConvASREncoder, ECAPAEncoder, SpeakerDecoder
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy


class Model(torch.nn.Module):
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
        encoder_target = encoder.pop('_target_')

        decoder = d['decoder'].copy()
        decoder.pop('_target_')

        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preprocessor)
        if 'ECAPAEncoder' in encoder_target:
            self.encoder = ECAPAEncoder(**encoder)
        else:
            self.encoder = ConvASREncoder(**encoder)
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
