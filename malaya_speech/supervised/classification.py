from malaya_boilerplate.huggingface import download_files
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
