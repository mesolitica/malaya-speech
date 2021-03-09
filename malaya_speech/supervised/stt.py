from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.utils.tf_featurization import STTFeaturizer
from malaya_speech.utils.subword import load as subword_load
from malaya_speech.model.tf import Transducer
from malaya_speech.path import TRANSDUCER_VOCAB
import json


def transducer_load(model, module, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': TRANSDUCER_VOCAB},
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

    input_nodes = [
        'X_placeholder',
        'X_len_placeholder',
        'encoded_placeholder',
        'predicted_placeholder',
        'states_placeholder',
    ]
    output_nodes = [
        'encoded',
        'ytu',
        'new_states',
        'padded_features',
        'padded_lens',
        'initial_states',
    ]
    if model in ['small-conformer-v2', 'conformer-v2', 'alconformer-v2']:
        output_nodes.append('greedy_decoder')

    inputs = {n: g.get_tensor_by_name(f'import/{n}:0') for n in input_nodes}
    outputs = {n: g.get_tensor_by_name(f'import/{n}:0') for n in output_nodes}

    return Transducer(
        X_placeholder = inputs['X_placeholder'],
        X_len_placeholder = inputs['X_len_placeholder'],
        encoded_placeholder = inputs['encoded_placeholder'],
        predicted_placeholder = inputs['predicted_placeholder'],
        states_placeholder = inputs['states_placeholder'],
        padded_features = outputs['padded_features'],
        padded_lens = outputs['padded_lens'],
        encoded = outputs['encoded'],
        ytu = outputs['ytu'],
        new_states = outputs['new_states'],
        initial_states = outputs['initial_states'],
        greedy = outputs.get('greedy_decoder'),
        featurizer = featurizer,
        vocab = vocab,
        time_reduction_factor = time_reduction_factor.get(model, 4),
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )
