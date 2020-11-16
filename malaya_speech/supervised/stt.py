from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.utils.tf_featurization import STTFeaturizer
from malaya_speech.model.tf import STT
import json


def load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)

    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)

    with open(path[model]['vocab']) as fopen:
        vocab = json.load(fopen)

    featurizer = STTFeaturizer(normalize_per_feature = True)

    return STT(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        X_len = g.get_tensor_by_name('import/Placeholder_1:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        seq_lens = g.get_tensor_by_name('import/seq_lens:0'),
        featurizer = featurizer,
        vocab = vocab,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )
