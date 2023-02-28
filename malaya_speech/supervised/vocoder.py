from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_boilerplate.huggingface import download_files
from malaya_speech.model.synthesis import Vocoder
from malaya_speech.torch_model.synthesis import Vocoder as PTVocoder


def load(model, module, quantized=False, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb'},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    inputs = ['Placeholder']
    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return Vocoder(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
    )


def load_pt_hifigan(model, **kwargs):
    s3_file = {
        'model': 'g_02500000',
        'config': 'config.json',
    }
    path = download_files(model, s3_file, **kwargs)
    return PTVocoder(
        pth=path['model'],
        config=path['config'],
        model=model,
        name='vocoder-hifigan',
        **kwargs,
    )
