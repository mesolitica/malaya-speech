from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.model.tf import UNET


def load(path, s3_path, model, name, **kwargs):
    check_file(path[model], s3_path[model], **kwargs)
    g = load_graph(path[model]['model'], **kwargs)

    return UNET(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )
