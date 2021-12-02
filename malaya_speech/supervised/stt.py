from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.utils.read import load as load_wav
from malaya_speech.utils.subword import load as subword_load
from malaya_speech.utils.tf_featurization import STTFeaturizer
from malaya_speech.model.tf import Transducer, Wav2Vec2_CTC, TransducerAligner
from malaya_speech.path import TRANSDUCER_VOCABS, TRANSDUCER_MIXED_VOCABS
import json
import os


def get_vocab(language):
    return TRANSDUCER_VOCABS.get(language, TRANSDUCER_VOCABS['malay'])


def get_vocab_mixed(language):
    return TRANSDUCER_MIXED_VOCABS.get(language)


dummy_sentences = ['tangan aku disentuh lembut', 'sebut perkataan angka']
time_reduction_factor = {
    'tiny-conformer': 4,
    'small-conformer': 4,
    'conformer': 4,
    'large-conformer': 4,
    'alconformer': 4,
}


def transducer_load(model, module, languages, quantized=False, stt=True, **kwargs):
    splitted = model.split('-')
    stack = False
    if len(splitted) == 3:
        stack = 'stack' == splitted[-2]
    if stack:
        keys = {'model': 'model.pb'}
        for no, language in enumerate(languages):
            keys[f'vocab_{no}'] = get_vocab_mixed(language)
        path = check_file(
            file=model,
            module=module,
            keys=keys,
            quantized=quantized,
            **kwargs,
        )
        vocab = []
        for no, language in enumerate(languages):
            vocab.append(subword_load(path[f'vocab_{no}'].replace('.subwords', '')))
    else:
        path = check_file(
            file=model,
            module=module,
            keys={'model': 'model.pb', 'vocab': get_vocab(splitted[-1])},
            quantized=quantized,
            **kwargs,
        )
        vocab = subword_load(path['vocab'].replace('.subwords', ''))
    g = load_graph(path['model'], **kwargs)
    featurizer = STTFeaturizer(normalize_per_feature=True)

    if stt:
        inputs = [
            'X_placeholder',
            'X_len_placeholder',
            'encoded_placeholder',
            'predicted_placeholder',
            'states_placeholder',
        ]
        outputs = [
            'encoded',
            'ytu',
            'new_states',
            'padded_features',
            'padded_lens',
            'initial_states',
            'greedy_decoder',
            'non_blank_transcript',
            'non_blank_stime',
        ]
        selected_model = Transducer
    else:
        inputs = [
            'X_placeholder',
            'X_len_placeholder',
            'subwords',
            'subwords_lens'
        ]
        outputs = [
            'padded_features',
            'padded_lens',
            'non_blank_transcript',
            'non_blank_stime',
            'decoded',
            'alignment'
        ]
        selected_model = TransducerAligner

    input_nodes, output_nodes = nodes_session(g, inputs, outputs)
    this_dir = os.path.dirname(__file__)
    wav1, _ = load_wav(os.path.join(this_dir, 'speech', '1.wav'))
    wav2, _ = load_wav(os.path.join(this_dir, 'speech', '2.wav'))

    return selected_model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        featurizer=featurizer,
        vocab=vocab,
        time_reduction_factor=time_reduction_factor.get(model, 4),
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
        wavs=[wav1, wav2],
        dummy_sentences=dummy_sentences,
        stack=stack,
    )


def wav2vec_transducer_load(model, module, quantized=False, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': get_vocab(model.split('-')[-1])},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    vocab = subword_load(path['vocab'].replace('.subwords', ''))

    inputs = [
        'X_placeholder',
        'X_len_placeholder',
        'encoded_placeholder',
        'predicted_placeholder',
        'states_placeholder',
    ]
    outputs = [
        'encoded',
        'ytu',
        'new_states',
        'padded_features',
        'padded_lens',
        'initial_states',
        'greedy_decoder',
    ]
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return Wav2Vec2_Transducer(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        vocab=vocab,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
    )


def wav2vec2_ctc_load(model, module, quantized=False, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.pb',
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    inputs = ['X_placeholder', 'X_len_placeholder']
    outputs = ['logits', 'seq_lens']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return Wav2Vec2_CTC(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
    )
