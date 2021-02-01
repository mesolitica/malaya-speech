from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import UNET, UNETSTFT, UNET1D


def load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder']
    outputs = ['logits']
    eager_g, input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNET(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        sess = generate_session(graph = g, **kwargs),
        eager_g = eager_g,
        model = model,
        name = name,
    )


def load_stft(
    path, s3_path, model, name, instruments, quantized = False, **kwargs
):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder']
    outputs = [f'logits_{i}' for i in range(len(instruments))]
    eager_g, input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNETSTFT(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        instruments = instruments,
        sess = generate_session(graph = g, **kwargs),
        eager_g = eager_g,
        model = model,
        name = name,
    )


def load_1d(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder']
    outputs = ['logits']
    eager_g, input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNET1D(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        sess = generate_session(graph = g, **kwargs),
        eager_g = eager_g,
        model = model,
        name = name,
    )
