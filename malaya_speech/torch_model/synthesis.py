import torch
import json
from malaya_speech.model.frame import Frame
from malaya_speech.torch_model.vits.commons import intersperse
from malaya_speech.torch_model.vits.model_infer import SynthesizerTrn
from malaya_speech.torch_model.vits import SID
from malaya_speech.torch_model.hifivoice.models import Generator
from malaya_speech.torch_model.hifivoice.env import AttrDict
from malaya_speech.torch_model.hifivoice.meldataset import mel_spectrogram
from malaya_speech.utils.text import TTS_SYMBOLS
from malaya_speech.utils.torch_utils import to_tensor_cuda, to_numpy
from malaya_speech.model.abstract import TTS

try:
    from malaya_boilerplate.hparams import HParams
except BaseException:
    from malaya_boilerplate.train.config import HParams


class VITS(SynthesizerTrn, TTS):
    def __init__(self, normalizer, pth, config, model, name, **kwargs):

        with open(config) as fopen:
            hps = HParams(**json.load(fopen))
        self.hps = hps

        TTS.__init__(self, e2e=True)
        super(VITS, self).__init__(
            len(TTS_SYMBOLS),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        self.eval()
        self.load_state_dict(torch.load(pth, map_location='cpu'))

        self._normalizer = normalizer
        self.__model__ = model
        self.__name__ = name

    def list_sid(self):
        """
        List available speakers for multispeaker model.
        """
        if self.n_speakers < 1:
            raise ValueError('this model is not multispeaker.')

        return SID.get(self.__model__, {})

    def predict(
        self,
        string,
        temperature: float = 0.0,
        temperature_durator: float = 0.0,
        length_ratio: float = 1.0,
        sid: int = None,
        **kwargs,
    ):
        """
        Change string to waveform.

        Parameters
        ----------
        string: str
        temperature: float, optional (default=0.0)
            Decoder model trying to decode with encoder(text) + random.normal() * temperature.
            Manipulate this variable will change speaking style.
        temperature_durator: float, optional (default=0.0)
            Durator trying to predict alignment with random.normal() * temperature_durator.
            Manipulate this variable will change speaking style.
        length_ratio: float, optional (default=1.0)
            Manipulate this variable will change length frames generated.
        sid: int, optional (default=None)
            speaker id, only available for multispeaker models.
            will throw an error if sid is None for multispeaker models.

        Returns
        -------
        result: Dict[string, ids, alignment, y]
        """
        if self.n_speakers > 0 and sid is None:
            raise ValueError('`sid` cannot be None for multispeaker model.')

        cuda = next(self.parameters()).is_cuda
        t, ids = self._normalizer.normalize(string, **kwargs)
        if self.hps.data.add_blank:
            ids = intersperse(ids, 0)
        ids = torch.LongTensor(ids)
        ids_lengths = torch.LongTensor([ids.size(0)])
        ids = ids.unsqueeze(0)
        ids = to_tensor_cuda(ids, cuda)
        ids_lengths = to_tensor_cuda(ids_lengths, cuda)
        if sid is not None:
            sid = torch.tensor([sid])
            sid = to_tensor_cuda(sid, cuda)
        audio = self.infer(
            ids,
            ids_lengths,
            noise_scale=temperature,
            noise_scale_w=temperature_durator,
            length_scale=length_ratio,
            sid=sid,
        )
        alignment = to_numpy(audio[1])[0, 0]
        audio = to_numpy(audio[0])[0, 0]
        return {
            'string': t,
            'ids': to_numpy(ids)[0],
            'alignment': alignment,
            'y': audio,
        }

    def forward(self, input, **kwargs):
        return self.predict(input, **kwargs)


class Vocoder(Generator):
    def __init__(self, pth, config, model, name, remove_weight_norm=False, **kwargs):
        with open(config) as fopen:
            json_config = json.load(fopen)

        self.h = AttrDict(json_config)

        super(Vocoder, self).__init__(self.h)

        self.state_dict_g = torch.load(pth, map_location='cpu')
        self.load_state_dict(self.state_dict_g['generator'])

        self.eval()
        if remove_weight_norm:
            self.remove_weight_norm()

        self.__model__ = model
        self.__name__ = name

    def get_mel(self, x):

        cuda = next(self.parameters()).is_cuda
        wav = torch.FloatTensor(x)
        wav = to_tensor_cuda(wav, cuda)

        return mel_spectrogram(
            wav.unsqueeze(0),
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax,
        )

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
        cuda = next(self.parameters()).is_cuda

        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        results = []
        with torch.no_grad():
            for input in inputs:
                x = torch.FloatTensor(input)
                x = to_tensor_cuda(x, cuda)
                y_g_hat = self.forward(x)
                audio = y_g_hat.squeeze()
                results.append(to_numpy(audio))

        return results
