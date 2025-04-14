from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
import logging

linear_cross_entropy_is_available = False
try:
    from cut_cross_entropy import linear_cross_entropy
    linear_cross_entropy_is_available = True
except:
    logging.warning('cut_cross_entropy is not available, peak memory for loss gradient will be high.')

class LlamaTTS(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config.num_codebooks = getattr(config, 'num_codebooks', 9)
        self.config.codebook_size = getattr(config, 'codebook_size', 1024)
        self.num_codebooks = self.config.num_codebooks
        self.codebook_size = self.config.codebook_size
        self.hidden_size = config.hidden_size
        self.codebook_heads = nn.Linear(self.hidden_size, self.hidden_size * config.num_codebooks, bias=False)
        
    def forward(self, input_ids, position_ids = None, attention_mask = None, labels = None, **kwargs):
        bsz, num_codebooks, seq_len = input_ids.shape
        inputs_embeds = self.model.embed_tokens(input_ids).sum(dim = 1)
        out_model = self.model(
            inputs_embeds = inputs_embeds, 
            position_ids = position_ids,
            attention_mask = attention_mask,
        )
        hidden_states = out_model.last_hidden_state
        codebook_hidden_states = self.codebook_heads(hidden_states).view(hidden_states.shape[0], -1, self.num_codebooks, self.hidden_size).transpose(1,2)
        
        if labels is not None:
            if linear_cross_entropy_is_available:
                loss = linear_cross_entropy(codebook_hidden_states, self.lm_head.weight, labels, shift = True)
            else:
                logits = model.lm_head(embedding)
                cross_entropy = CrossEntropyLoss(reduction='mean')
                logits = logits[:,:,:-1].reshape(-1, logits.shape[-1])
                labels = labels[:,:,1:].reshape(-1)
                loss = cross_entropy(logits, labels)
            return loss / self.num_codebooks
        else:
            return codebook_hidden_states

