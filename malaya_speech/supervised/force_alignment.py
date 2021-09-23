from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)


def load(model, module, quantized=False, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb'},
        quantized=quantized,
        **kwargs,
    )
