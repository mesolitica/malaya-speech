from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.model.tf import Vocoder


def load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)

    return Vocoder(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )
