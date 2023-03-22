import torch
import numpy as np
from malaya_speech.model.frame import Frame
from malaya_speech.utils.subword import (
    SentencePieceTokenProcessor,
    merge_sentencepiece_tokens,
)
from malaya_speech.utils.torch_featurization import (
    FeatureExtractor,
    RNNTBeamSearch,
    post_process_hypos,
    conformer_rnnt_base,
    conformer_rnnt_tiny,
    conformer_rnnt_medium,
    conformer_rnnt_large,
    emformer_rnnt_base,
)
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy

model_mapping = {
    'mesolitica/conformer-base': conformer_rnnt_base,
    'mesolitica/conformer-tiny': conformer_rnnt_tiny,
    'mesolitica/conformer-medium': conformer_rnnt_medium,
    'mesolitica/conformer-medium-mixed': conformer_rnnt_medium,
    'mesolitica/conformer-base-singlish': conformer_rnnt_base,
    'mesolitica/emformer-base': emformer_rnnt_base,
    'mesolitica/conformer-medium-mixed-augmented': conformer_rnnt_medium,
    'mesolitica/conformer-large-mixed-augmented': conformer_rnnt_large,
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


class ForceAlignment(Conformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, input, transcription: str, temperature: float = 1.0):
        """
        Transcribe input, will return a string.

        Parameters
        ----------
        input: np.array
            np.array or malaya_speech.model.frame.Frame.
        transcription: str
            transcription of input audio
        temperature: float, optional (default=1.0)
            temperature for logits.

        Returns
        -------
        result: Dict[words_alignment, subwords_alignment, subwords, alignment]
        """
        cuda = next(self.parameters()).is_cuda
        input = input.array if isinstance(input, Frame) else input
        len_input = len(input)
        mel, mel_len = self.feature_extractor(input)
        input, length = mel, mel_len
        if input.dim() != 2 and not (input.dim() == 3 and input.shape[0] == 1):
            raise ValueError("input must be of shape (T, D) or (1, T, D)")
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if length.shape != () and length.shape != (1,):
            raise ValueError("length must be of shape () or (1,)")
        if input.dim() == 0:
            input = input.unsqueeze(0)
        enc_out, _ = self.model.transcribe(input, length)

        phonemes = self.tokenizer.sp_model.encode(transcription.lower())
        blank_idx = self.tokenizer.sp_model.get_piece_size()

        prediction, alignment = [], []

        with torch.no_grad():
            token = blank_idx
            state = None
            time = 0
            time_phoneme = 0
            total = enc_out.shape[1]
            total_phoneme = len(phonemes)
            one_tensor = to_tensor_cuda(torch.tensor([1]), cuda)
            pred_out, _, pred_state = self.model.predict(
                to_tensor_cuda(torch.tensor([[token]]), cuda),
                one_tensor,
                state)

            hypothesis = [blank_idx, pred_state]
            while time < total and time_phoneme < total_phoneme:

                token = hypothesis[0]
                state = hypothesis[1]
                pred_out, _, pred_state = self.model.predict(
                    to_tensor_cuda(torch.tensor([[token]]), cuda),
                    one_tensor,
                    state)
                joined_out, _, _ = self.model.join(
                    enc_out[:, time: time + 1],
                    one_tensor,
                    pred_out,
                    to_tensor_cuda(torch.tensor([1]), cuda),
                )
                joined_out = torch.nn.functional.log_softmax(
                    joined_out / temperature, dim=3)[:, 0, 0]
                _predict = joined_out.argmax(-1)[0]
                _equal = _predict == blank_idx
                if _equal:
                    _predict = blank_idx
                    _index = hypothesis[0]
                    _states = hypothesis[1]

                else:
                    _predict = phonemes[time_phoneme]
                    _index = _predict
                    _states = pred_state
                    time_phoneme += 1

                hypothesis = [_index, _states]
                prediction.append(_predict)
                alignment.append(joined_out[0])

                time += 1

        skip = len_input / enc_out.shape[1]
        aranged = np.arange(0, len_input, skip)

        alignment = np.exp(to_numpy(torch.stack(alignment)))
        alignments = []
        for i, p in enumerate(prediction):
            if p != blank_idx:
                alignments.append(alignment[i])
        alignments = np.stack(alignments)[:, [p for p in prediction if p != blank_idx]]

        decoded = [self.tokenizer.sp_model.IdToPiece(
            [p])[0] if p != blank_idx else None for p in prediction]
        subwords_alignment = []
        for i in range(len(decoded)):
            if decoded[i]:
                data = {
                    'text': decoded[i],
                    'start': aranged[i] / self.sample_rate,
                    'end': (aranged[i] / self.sample_rate) + (skip / self.sample_rate),
                }
                subwords_alignment.append(data)

        bpes = [(s['text'], s) for s in subwords_alignment]
        merged_bpes = merge_sentencepiece_tokens(bpes)
        words_alignment = []
        for m in merged_bpes:
            if isinstance(m[1], list):
                start = m[1][0]['start']
                end = m[1][-1]['end']
            else:
                start = m[1]['start']
                end = m[1]['end']
            words_alignment.append({
                'text': m[0].replace('▁', ''),
                'start': start,
                'end': end,
            })

        t = (len_input / self.sample_rate) / alignments.shape[0]

        for i in range(len(words_alignment)):
            start = int(round(words_alignment[i]['start'] / t))
            end = int(round(words_alignment[i]['end'] / t))
            words_alignment[i]['start_t'] = start
            words_alignment[i]['end_t'] = end
            words_alignment[i]['score'] = alignments[start: end + 1, i].max()

        for i in range(len(subwords_alignment)):
            start = int(round(subwords_alignment[i]['start'] / t))
            end = int(round(subwords_alignment[i]['end'] / t))
            subwords_alignment[i]['start_t'] = start
            subwords_alignment[i]['end_t'] = end
            subwords_alignment[i]['score'] = alignments[start: end + 1, i].max()

        return {
            'words_alignment': words_alignment,
            'subwords_alignment': subwords_alignment,
            'subwords': [
                self.tokenizer.sp_model.IdToPiece(
                    [p])[0] for p in prediction if p != blank_idx],
            'alignment': alignments,
        }
