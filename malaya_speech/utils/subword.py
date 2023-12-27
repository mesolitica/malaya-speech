import re
import six
from typing import List
import logging

logger = logging.getLogger(__name__)

BLANK = 0

sentencepiece_available = False
try:
    import sentencepiece as spm
    sentencepiece_available = True
except Exception as e:
    logger.warning(
        '`sentencepiece` is not available, any models that use sentencepiece will not able to use.')


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
        filtered_hypo_tokens = [token_index for token_index in tokens[1:]
                                if token_index not in self.post_process_remove_list]
        output_string = ''.join(
            self.sp_model.id_to_piece(filtered_hypo_tokens)).replace(
            '\u2581', ' ')

        if lstrip:
            return output_string.lstrip()
        else:
            return output_string


def merge_sentencepiece_tokens(
    paired_tokens,
    **kwargs,
):
    new_paired_tokens = []
    n_tokens = len(paired_tokens)

    i = 0

    while i < n_tokens:

        current_token, current_weight = paired_tokens[i]
        if isinstance(current_token, bytes):
            current_token = current_token.decode()
        if not current_token.startswith('▁'):
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while (
                not current_token.startswith('▁')
            ):
                merged_token = merged_token + current_token.replace('▁', '')
                merged_weight.append(current_weight)
                i = i + 1
                if i < n_tokens:
                    current_token, current_weight = paired_tokens[i]
                else:
                    break
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1

    return new_paired_tokens


def merge_bpe_tokens(
    paired_tokens,
    rejected=['<s>', '</s>', '<unk>', '<pad>', '<mask>'],
    prefix_char='Ġ',
    **kwargs,
):
    new_paired_tokens = []
    paired_tokens = [t for t in paired_tokens if t[0] not in rejected]
    n_tokens = len(paired_tokens)

    i = 0

    while i < n_tokens:

        current_token, current_weight = paired_tokens[i]
        if isinstance(current_token, bytes):
            current_token = current_token.decode()
        if i > 0 and not current_token.startswith(prefix_char) and current_token not in rejected:
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while (
                not current_token.startswith(prefix_char)
                and current_token not in rejected
            ):
                merged_token = merged_token + current_token.replace(prefix_char, '')
                merged_weight.append(current_weight)
                i = i + 1
                if i < n_tokens:
                    current_token, current_weight = paired_tokens[i]
                else:
                    break
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1

    words = [
        i[0].replace(prefix_char, '') for i in new_paired_tokens if i[0] not in rejected
    ]
    weights = [i[1] for i in new_paired_tokens if i[0] not in rejected]
    return list(zip(words, weights))
