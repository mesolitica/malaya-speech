from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.model.tf import (
    Speakernet,
    Speaker2Vec,
    SpeakernetClassification,
    Classification,
)
from malaya_speech.utils import featurization
from malaya_speech.config import (
    speakernet_featurizer_config as speakernet_config,
)


def load(path, s3_path, model, name, extra, label, quantized = False, **kwargs):

    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)

    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)

    vectorizer_mapping = {
        'vggvox-v1': featurization.vggvox_v1,
        'vggvox-v2': featurization.vggvox_v2,
        'deep-speaker': featurization.deep_speaker,
        'speakernet': featurization.SpeakerNetFeaturizer(
            **{**speakernet_config, **extra}
        ),
    }

    if name == 'speaker-vector':
        if model == 'speakernet':
            model_class = Speakernet
        else:
            model_class = Speaker2Vec
    else:
        if model == 'speakernet':
            model_class = SpeakernetClassification
        else:
            model_class = Classification

    if model == 'speakernet':
        return model_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            X_len = g.get_tensor_by_name('import/Placeholder_1:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = vectorizer_mapping[model],
            sess = generate_session(graph = g, **kwargs),
            model = model,
            extra = extra,
            label = label,
            name = name,
        )

    else:
        return model_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = vectorizer_mapping[model],
            sess = generate_session(graph = g, **kwargs),
            model = model,
            extra = extra,
            label = label,
            name = name,
        )
