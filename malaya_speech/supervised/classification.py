from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.model.tf import CLASSIFICATION
from malaya_speech import featurization


def load(path, s3_path, model, name, extra, **kwargs):
    check_file(path[model], s3_path[model], **kwargs)
    g = load_graph(path[model]['model'], **kwargs)

    vectorizer_mapping = {
        'vggvox-v1': featurization.vggvox_v1,
        'vggvox-v2': featurization.vggvox_v2,
    }

    return CLASSIFICATION(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        vectorizer = vectorizer_mapping[model],
        sess = generate_session(graph = g, **kwargs),
        model = model,
        extra = extra,
        name = name,
    )
