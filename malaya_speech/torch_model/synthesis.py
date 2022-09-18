import torch
import json
from malaya_boilerplate.train.config import HParams
from malaya_speech.torch_model.vits.commons import intersperse
from malaya_speech.torch_model.vits.model_infer import SynthesizerTrn
from malaya_speech.utils.text import TTS_SYMBOLS
from malaya_speech.utils.torch_utils import to_tensor_cuda, to_numpy
from malaya_speech.model.abstract import Abstract, TTS


class VITS(SynthesizerTrn, TTS):
    def __init__(self, normalizer, pth, config, model, name, **kwargs):

        with open(config) as fopen:
            hps = HParams(**json.load(fopen))
        self.hps = hps

        TTS.__init__(self, e2e=True)
        super(VITS, self).__init__(len(TTS_SYMBOLS),
                                   hps.data.filter_length // 2 + 1,
                                   hps.train.segment_size // hps.data.hop_length,
                                   **hps.model)
        self.eval()
        self.load_state_dict(torch.load(pth))

        self._normalizer = normalizer
        self.__model__ = model
        self.__name__ = name

    def predict(
        self,
        string,
        temperature: float = 0.6666,
        temperature_durator: float = 0.6666,
        length_ratio: float = 1.0,
        **kwargs,
    ):
        """
        Change string to waveform.

        Parameters
        ----------
        string: str
        temperature: float, optional (default=0.6666)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
        temperature_durator: float, optional (default=0.6666)
            Durator trying to predict alignment with random.normal() * temperature_durator.
        length_ratio: float, optional (default=1.0)
            Increase this variable will increase time voice generated.

        Returns
        -------
        result: Dict[string, ids, alignment, y]
        """
        cuda = next(self.parameters()).is_cuda
        t, ids = self._normalizer.normalize(string, **kwargs)
        if self.hps.data.add_blank:
            ids = intersperse(ids, 0)
        ids = torch.LongTensor(ids)
        ids_lengths = torch.LongTensor([ids.size(0)])
        ids = ids.unsqueeze(0)
        ids = to_tensor_cuda(ids, cuda)
        ids_lengths = to_tensor_cuda(ids_lengths, cuda)
        audio = self.infer(
            ids,
            ids_lengths,
            noise_scale=temperature,
            noise_scale_w=temperature_durator,
            length_scale=length_ratio,
        )
        alignment = to_numpy(audio[1])[0, 0]
        audio = to_numpy(audio[0])[0, 0]
        return {
            'string': t,
            'ids': to_numpy(ids)[0],
            'alignment': alignment,
            'y': audio,
        }

    def __call__(self, input, **kwargs):
        return self.predict(input, **kwargs)
