from typing import List
import logging

logger = logging.getLogger(__name__)

sentencepiece_available = False
try:
    import sentencepiece as spm
    sentencepiece_available = True
except Exception as e:
    logger.warning('`sentencepiece` is not available, any models that use sentencepiece will not able to use.')


def load_sentencepiece(model_file):
    """
    Parameters
    ----------
    model_file: str
        sentencepiece model file.

    Returns
    --------
    result: sentencepiece.SentencePieceProcessor
    """

    if not sentencepiece_available:
        raise ModuleNotFoundError(
            'sentencepiece not installed. Please install it by `pip install sentencepiece` and try again.'
        )
    return spm.SentencePieceProcessor(model_file=model_file)


class SentencePieceTokenProcessor:
    def __init__(self, sp_model_path):
        self.sp_model = load_sentencepiece(model_file=sp_model_path)
        self.post_process_remove_list = {
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        }

    def __call__(self, tokens: List[int], lstrip: bool = True) -> str:
        filtered_hypo_tokens = [
            token_index for token_index in tokens[1:] if token_index not in self.post_process_remove_list
        ]
        output_string = ''.join(self.sp_model.id_to_piece(filtered_hypo_tokens)).replace('\u2581', ' ')

        if lstrip:
            return output_string.lstrip()
        else:
            return output_string
