import torch.nn as nn
import torch
from malaya_speech.model.frame import Frame
from malaya_speech.utils.torch_utils import to_tensor_cuda, to_numpy, from_log

try:
    import voicefixer
except Exception as e:
    raise ModuleNotFoundError(
        'voicefixer not installed. Please install it by `pip3 install voicefixer==0.0.18` and try again.'
    )


class VoiceFixer(voicefixer.VoiceFixer):
    def __init__(self):
        super(VoiceFixer, self).__init__()

    def predict(self, input, remove_higher_frequency: bool = True):
        """
        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame,
            must an audio with 44100 sampling rate.
        remove_higher_frequency: bool, optional (default = True)
            Remove high frequency before neural upsampling.

        Returns
        -------
        result: np.array
        """

        input = input.array if isinstance(input, Frame) else input
        wav_10k = input
        cuda = next(self.parameters()).is_cuda

        res = []
        seg_length = 44100 * 30
        break_point = seg_length
        while break_point < wav_10k.shape[0] + seg_length:
            segment = wav_10k[break_point - seg_length: break_point]
            if remove_higher_frequency:
                segment = self.remove_higher_frequency(segment)

            sp, mel_noisy = self._pre(self._model, segment, cuda)
            out_model = self._model(sp, mel_noisy)
            denoised_mel = from_log(out_model['mel'])
            out = self._model.vocoder(denoised_mel, cuda)

            if torch.max(torch.abs(out)) > 1.0:
                out = out / torch.max(torch.abs(out))

            out, _ = self._trim_center(out, segment)
            res.append(out)
            break_point += seg_length

        out = torch.cat(res, -1)
        return to_numpy(out[0][0])

    def forward(self, input, remove_higher_frequency: bool = True):
        return self.predict(input=input, remove_higher_frequency=remove_higher_frequency)
