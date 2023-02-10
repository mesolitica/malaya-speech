import torch
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
import logging

logger = logging.getLogger(__name__)

LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)


class LM(torch.nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        alpha: float = 0.05,
        beta: float = 0.5,
        order: int = 1,
        **kwargs,
    ):
        super().__init__()
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|pad|>']})
        tokenizer.pad_token = '<|pad|>'
        model.resize_token_embeddings(len(tokenizer))
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.order = order

    def score_partial_token(self, partial_token: str):
        return 0.0

    def get_start_state(self):
        cuda = next(self.model.parameters()).is_cuda
        tokenized = self.tokenizer.encode(self.tokenizer.bos_token, return_tensors='pt')
        past_key_values = None
        with torch.no_grad():
            context = to_tensor_cuda(torch.tensor(tokenized)[0], cuda)
            output = self.model(context, past_key_values=past_key_values)
        sent_logits = output.logits
        sent_logits[:, self.tokenizer.pad_token_id] = float('-inf')
        return {'current_score': float('-inf'),
                'last_state': output.past_key_values,
                'score': sent_logits.logsumexp(1),
                'last_score': None,
                }

    def score(self, prev_state, word: str, is_last_word: bool = False):
        cuda = next(self.model.parameters()).is_cuda
        tokenized = self.tokenizer.encode(word, return_tensors='pt')[0]
        with torch.no_grad():
            context = to_tensor_cuda(torch.tensor(tokenized), cuda)
            output = self.model(context, past_key_values=prev_state['last_state'])
        sent_logits = output.logits
        sent_logits[:, self.tokenizer.pad_token_id] = float('-inf')
        score = sent_logits.gather(1, tokenized.unsqueeze(1)).squeeze(1)
        if prev_state['last_score'] is not None:
            s = torch.concat([prev_state['last_score'], score])
        else:
            s = score

        if len(sent_logits[:-1]):
            c = torch.concat([prev_state['score'], sent_logits[:-1].logsumexp(1)])
        else:
            c = prev_state['score']
        score_ = s - c

        current_score = score_.sum()

        logger.debug(f'word: {word}, current_score: {current_score}')

        new_state = {'current_score': current_score,
                     'last_state': output.past_key_values,
                     'score': torch.concat([prev_state['score'], sent_logits.logsumexp(1)]),
                     'last_score': s,
                     }
        lm_score = self.alpha * new_state['current_score'] * LOG_BASE_CHANGE_FACTOR + self.beta
        return float(to_numpy(lm_score)), new_state
