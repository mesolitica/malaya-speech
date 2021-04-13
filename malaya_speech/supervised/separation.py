from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import Split_Wav, Split_Mel


def load(model, module, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb'},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    outputs = ['logits']

    if module.split('-')[-1] == 'wav':
        selected_class = Split_Wav
        inputs = ['Placeholder']
    else:
        selected_class = Split_Mel
        inputs = ['Placeholder', 'Placeholder_1']

    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return selected_class(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = module,
    )
