from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.utils.tf_featurization import STTFeaturizer
from malaya_speech.utils.subword import load as subword_load
from malaya_speech.model.tf import STT, Transducer
import json


def ctc_load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)

    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)

    with open(path[model]['vocab']) as fopen:
        vocab = json.load(fopen) + ['{', '}', '[']

    featurizer = STTFeaturizer(normalize_per_feature = True)

    return STT(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        X_len = g.get_tensor_by_name('import/Placeholder_1:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        seq_lens = g.get_tensor_by_name('import/seq_lens:0'),
        featurizer = featurizer,
        vocab = vocab,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )


def transducer_load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)

    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    vocab = subword_load(path[model]['vocab'].replace('.subwords', ''))
    featurizer = STTFeaturizer(normalize_per_feature = True)

    time_reduction_factor = {'small-conformer': 4, 'conformer': 4}

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
        featurizer = featurizer,
        vocab = vocab,
        time_reduction_factor = time_reduction_factor[model],
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )
