from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.utils.tf_featurization import STTFeaturizer
from malaya_speech.utils.subword import load as subword_load
from malaya_speech.model.tf import Transducer
from malaya_speech.path import TRANSDUCER_VOCABS
import json


def get_vocab(language):
    return TRANSDUCER_VOCABS.get(language, TRANSDUCER_VOCABS['malay'])


def transducer_load(model, module, quantized = False, **kwargs):

    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': get_vocab(model.split('-')[-1])},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    vocab = subword_load(path['vocab'].replace('.subwords', ''))
    featurizer = STTFeaturizer(normalize_per_feature = True)

    time_reduction_factor = {
        'small-conformer': 4,
        'conformer': 4,
        'large-conformer': 4,
        'alconformer': 4,
    }

    inputs = [
        'X_placeholder',
        'X_len_placeholder',
        'encoded_placeholder',
        'predicted_placeholder',
        'states_placeholder',
    ]
    outputs = [
        'encoded',
        'ytu',
        'new_states',
        'padded_features',
        'padded_lens',
        'initial_states',
        'greedy_decoder',
        'non_blank_transcript',
        'non_blank_stime',
    ]
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return Transducer(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        featurizer = featurizer,
        vocab = vocab,
        time_reduction_factor = time_reduction_factor.get(model, 4),
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )
