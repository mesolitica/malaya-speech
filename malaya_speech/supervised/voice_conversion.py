from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import FastVC
from malaya_speech import speaker_vector


def load(model, module, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb'},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    inputs = ['mel', 'ori_vector', 'target_vector', 'mel_lengths']
    outputs = ['mel_before', 'mel_after']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    speaker_vector_model = '-'.join(model.split('-')[2:])

    speaker_model = speaker_vector.deep_model(speaker_vector_model, **kwargs)

    magnitudes = {
        'vggvox-v2': lambda x: x * 30 - 3.5,
        'speakernet': lambda x: x * 3,
    }

    return FastVC(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        speaker_vector = speaker_model,
        magnitude = magnitude[speaker_vector_model],
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )
