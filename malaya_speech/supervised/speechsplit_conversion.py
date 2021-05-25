from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import FastSpeechSplit
from malaya_speech import speaker_vector, gender


def load(model, module, f0_mode = 'pyworld', quantized = False, **kwargs):
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
    if f0_mode == 'pysptk':
        gender_model = gender.deep_model(speaker_vector_model, **kwargs)
    else:
        gender_model = None

    return FastSpeechSplit(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        speaker_vector = speaker_model,
        gender_model = gender_model,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )
