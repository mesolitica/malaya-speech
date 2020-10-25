from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.model.tf import SPEAKER2VEC, SPEAKERNET, CLASSIFICATION
from malaya_speech.utils import featurization


def load(path, s3_path, model, name, extra, label, **kwargs):

    import malaya_speech.config

    speakernet_config = malaya_speech.config.speakernet_featurizer_config

    check_file(path[model], s3_path[model], **kwargs)
    g = load_graph(path[model]['model'], **kwargs)

    vectorizer_mapping = {
        'vggvox-v1': featurization.vggvox_v1,
        'vggvox-v2': featurization.vggvox_v2,
        'deep-speaker': featurization.read_mfcc,
        'speakernet': featurization.SpeakerNetFeaturizer(speakernet_config),
    }

    if name == 'speaker-vector':
        if model == 'speakernet':
            model_class = SPEAKERNET
        else:
            model_class = SPEAKER2VEC
    else:
        model_class = CLASSIFICATION

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
