from malaya_boilerplate.huggingface import download_files
from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.classification import (
    Speakernet,
    Speaker2Vec,
    Transformer2Vec,
    SpeakernetClassification,
    MarbleNetClassification,
    Classification,
)
from malaya_speech.utils import featurization
from malaya_speech.config import (
    speakernet_featurizer_config as speakernet_config,
)
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioXVector,
)
from malaya_speech.torch_model.huggingface import XVector
from malaya_speech.torch_model.nemo import (
    SpeakerVector as NemoSpeakerVector,
    Classification as NemoClassification,
)

transformer_models = ['conformer-tiny', 'conformer-base', 'vit-tiny', 'vit-base']


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
        elif model in transformer_models:
            model_class = Transformer2Vec
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
    elif 'marblenet' in model or model in transformer_models:
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


def huggingface_xvector(model, **kwargs):
    hf_model = AutoModelForAudioXVector.from_pretrained(model, **kwargs)
    processor = AutoFeatureExtractor.from_pretrained(model, **kwargs)

    return XVector(
        hf_model=hf_model,
        processor=processor,
        model=model,
        name='speaker-vector-huggingface',
    )


def nemo_speaker_vector(model, **kwargs):
    s3_file = {
        'config': 'model_config.yaml',
        'model': 'model_weights.ckpt',
    }
    path = download_files(model, s3_file, **kwargs)
    return NemoSpeakerVector(
        config=path['config'],
        pth=path['model'],
        model=model,
        name='speaker-vector-nemo',
    )


def nemo_classification(model, label, **kwargs):
    s3_file = {
        'config': 'model_config.yaml',
        'model': 'model_weights.ckpt',
    }
    path = download_files(model, s3_file, **kwargs)
    return NemoClassification(
        config=path['config'],
        pth=path['model'],
        label=label,
        model=model,
        name='classification-nemo',
    )
