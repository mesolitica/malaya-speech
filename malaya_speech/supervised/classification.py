from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.classification import (
    Speakernet,
    Speaker2Vec,
    SpeakernetClassification,
    MarbleNetClassification,
    Classification,
)
from malaya_speech.utils import featurization
from malaya_speech.config import (
    speakernet_featurizer_config as speakernet_config,
)


def load(model, module, extra, label, quantized=False, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb'},
        quantized=quantized,
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
        elif 'marblenet' in model:
            model_class = MarbleNetClassification
        else:
            model_class = Classification

    if model == 'speakernet':
        inputs = ['Placeholder', 'Placeholder_1']
    elif 'marblenet' in model:
        inputs = ['X_placeholder', 'X_len_placeholder']
    else:
        inputs = ['Placeholder']
    outputs = ['logits']

    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        vectorizer=vectorizer_mapping.get(model),
        sess=generate_session(graph=g, **kwargs),
        model=model,
        extra=extra,
        label=label,
        name=module,
    )
