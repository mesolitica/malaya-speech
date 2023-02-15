import torch
from malaya_speech.model.frame import Frame
from malaya_speech.utils.bpe import load_sentencepiece, SentencePieceTokenProcessor
from malaya_speech.utils.torch_featurization import (
    FeatureExtractor,
    RNNTBeamSearch,
    post_process_hypos,
    conformer_rnnt_base,
    conformer_rnnt_tiny,
    conformer_rnnt_medium,
    emformer_rnnt_base,
)
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy

model_mapping = {
    'mesolitica/conformer-base': conformer_rnnt_base,
    'mesolitica/conformer-tiny': conformer_rnnt_tiny,
    'mesolitica/conformer-medium': conformer_rnnt_medium,
    'mesolitica/emformer-base': emformer_rnnt_base,
}


class Conformer(torch.nn.Module):

    sample_rate = 16000
    segment_length = 16
    hop_length = 160
    right_context_length = 4

    def __init__(self, pth, sp_model, stats_file, model, name):
        super().__init__()

        conformer = model_mapping[model]()
        conformer.load_state_dict(torch.load(pth, map_location='cpu'))

        self.model = conformer
        self.tokenizer = SentencePieceTokenProcessor(sp_model)
        self.feature_extractor = FeatureExtractor(stats_file, pad='emformer' in model)

        self.blank_idx = self.tokenizer.sp_model.get_piece_size()
        self.decoder = RNNTBeamSearch(self.model, self.blank_idx)

        self.__model__ = model
        self.__name__ = name

        self.rnnt_streaming = 'emformer' in model

    def forward(self, inputs, beam_width: int = 20):
        """
        Transcribe inputs using beam decoder.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        beam_width: int, optional (default=20)
            beam size for beam decoder.

        Returns
        -------
        result: List[Tuple]
        """
        cuda = next(self.parameters()).is_cuda

        inputs = [
            input.array if isinstance(input, Frame) else input
            for input in inputs
        ]

        results = []
        for input in inputs:
            mel, mel_len = self.feature_extractor(input)
            mel = to_tensor_cuda(mel, cuda)
            mel_len = to_tensor_cuda(mel_len, cuda)

            hypotheses = self.decoder(mel, mel_len, beam_width)
            results.append(post_process_hypos(hypotheses, self.tokenizer.sp_model))

        return results

    def beam_decoder(self, inputs, beam_width: int = 20):
        """
        Transcribe inputs using beam decoder.

        Parameters
        ----------
        inputs: List[np.array]
            List[np.array] or List[malaya_speech.model.frame.Frame].
        beam_width: int, optional (default=20)
            beam size for beam decoder.

        Returns
        -------
        result: List[str]
        """
        r = self.forward(inputs=inputs, beam_width=beam_width)
        return [r_[0][0] for r_ in r]
