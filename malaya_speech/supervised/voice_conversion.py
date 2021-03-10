from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import FastVC
from malaya_speech.utils.featurization import universal_mel
from malaya_speech import speaker_vector


def load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['mel', 'ori_vector', 'target_vector', 'mel_lengths']
    outputs = ['mel_before', 'mel_after']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    speaker_model = speaker_vector.deep_model('vggvox-v2', **kwargs)
    return FastVC(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        waveform_to_mel = universal_mel,
        speaker_vector = speaker_model,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )
