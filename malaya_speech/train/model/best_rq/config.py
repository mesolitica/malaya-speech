from ..abstract import Dataclass
from typing import List, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class Best_RQConfig(Dataclass):
    extractor_mode: str = field(
        default='default',
        metadata={
            'help': 'mode for feature extractor. default has a single group norm with d '
            'groups in the first conv block, whereas layer_norm has layer norms in '
            'every block (meant to use with normalize=True)'
        },
    )
    encoder_embed_dim: int = field(
        default=768, metadata={'help': 'encoder embedding dimension'}
    )
    dropout: float = field(
        default=0.1,
        metadata={'help': 'dropout probability for the transformer'},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={'help': 'dropout probability for attention weights'},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={'help': 'dropout probability after activation in FFN'},
    )
    encoder_layerdrop: float = field(
        default=0.05,
        metadata={'help': 'probability of dropping a tarnsformer layer'},
    )
    dropout_input: float = field(
        default=0.1,
        metadata={'help': 'dropout to apply to the input (after feat extr)'},
    )
    dropout_features: float = field(
        default=0.1,
        metadata={
            'help': 'dropout to apply to the features (after feat extr)'
        },
    )
    final_dim: int = field(
        default=256,
        metadata={
            'help': 'project final representations and targets to this many dimensions.'
            'set to encoder_embed_dim is <= 0'
        },
    )
    conv_feature_layers: str = field(
        default='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]',
        metadata={
            'help': 'string describing convolutional feature extraction layers in form of a python list that contains '
            '[(dim, kernel_size, stride), ...]'},
    )
    conv_bias: bool = field(
        default=False, metadata={'help': 'include bias in conv encoder'}
    )
    embedding_dim: int = field(
        default=16,
        metadata={
            'help': 'embedding size for codebook'
        },
    )
    num_embeddings: int = field(
        default=8192,
        metadata={
            'help': 'size of embedding for codebook'
        },
    )

    mask_length: int = field(default=10, metadata={'help': 'mask length'})
    mask_prob: float = field(
        default=0.65,
        metadata={'help': 'probability of replacing a token with mask'},
    )
    mask_selection: str = field(
        default='static', metadata={'help': 'how to choose mask length'}
    )
    mask_other: float = field(
        default=0,
        metadata={
            'help': 'secondary mask argument (used for more complex distributions), '
            'see help in compute_mask_indices'
        },
    )
    no_mask_overlap: bool = field(
        default=False,
        metadata={'help': 'whether to allow masks to overlap'},
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            'help': 'min space between spans (if no overlap is enabled)'
        },
    )

    mask_channel_length: int = field(
        default=10,
        metadata={'help': 'length of the mask for features (channels)'},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={'help': 'probability of replacing a feature with 0'},
    )
    mask_channel_selection: str = field(
        default='static',
        metadata={'help': 'how to choose mask length for channel masking'},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            'help': 'secondary mask argument (used for more complex distributions), '
            'see help in compute_mask_indicesh'
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={'help': 'whether to allow channel masks to overlap'},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            'help': 'min space between spans (if no overlap is enabled)'
        },
    )
