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


def load(model, module, extra, label, quantized = False, **kwargs):

    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb'},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    vectorizer_mapping = {
        'vggvox-v1': featurization.vggvox_v1,
        'vggvox-v2': featurization.vggvox_v2,
        'deep-speaker': featurization.deep_speaker,
        'speakernet': featurization.SpeakerNetFeaturizer(
            **{**speakernet_config, **extra}
        ),
    }

    if module == 'speaker-vector':
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

    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes = input_nodes,
        output_nodes = output_nodes,
        vectorizer = vectorizer_mapping[model],
        sess = generate_session(graph = g, **kwargs),
        model = model,
        extra = extra,
        label = label,
        name = module,
    )
