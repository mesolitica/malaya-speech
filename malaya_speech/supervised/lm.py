from malaya_speech.utils import check_file
from malaya_speech.torch_model.gpt2_lm import LM
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def load(model, module, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.klm',
        },
        quantized=False,
        **kwargs,
    )
    return path['model']


def gpt2_load(model, **kwargs):
    tokenizer = GPT2Tokenizer.from_pretrained(model, use_fast=False, add_special_tokens=False)
    model = GPT2LMHeadModel.from_pretrained(model)
    return LM(model, tokenizer, **kwargs)
