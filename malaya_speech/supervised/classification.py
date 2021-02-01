from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
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
        inputs = ['Placeholder', 'Placeholder_1']
        outputs = ['logits']
    else:
        inputs = ['Placeholder']
        outputs = ['logits']

    eager_g, input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        vectorizer = vectorizer_mapping[model],
        sess = generate_session(graph = g, **kwargs),
        eager_g = eager_g,
        model = model,
        extra = extra,
        label = label,
        name = name,
    )
