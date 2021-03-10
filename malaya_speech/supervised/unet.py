from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import UNET, UNETSTFT, UNET1D


def load(model, module, quantized = False, **kwargs):

    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb'},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    inputs = ['Placeholder']
    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNET(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )


def load_stft(model, module, instruments, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb'},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    inputs = ['Placeholder']
    outputs = [f'logits_{i}' for i in range(len(instruments))]
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNETSTFT(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        instruments = instruments,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )


def load_1d(model, module, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb'},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    inputs = ['Placeholder']
    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return UNET1D(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )
