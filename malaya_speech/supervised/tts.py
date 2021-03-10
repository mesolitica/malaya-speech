from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import Tacotron, Fastspeech
import numpy as np


def tacotron_load(
    path, s3_path, model, name, normalizer, quantized = False, **kwargs
):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder', 'Placeholder_1']
    outputs = ['decoder_output', 'post_mel_outputs', 'alignment_histories']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    stats = np.load(path[model]['stats'])

    return Tacotron(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        normalizer = normalizer,
        stats = stats,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )


def fastspeech_load(
    path, s3_path, model, name, normalizer, quantized = False, **kwargs
):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder', 'speed_ratios', 'f0_ratios', 'energy_ratios']
    outputs = ['decoder_output', 'post_mel_outputs']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    stats = np.load(path[model]['stats'])

    return Fastspeech(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        normalizer = normalizer,
        stats = stats,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )
