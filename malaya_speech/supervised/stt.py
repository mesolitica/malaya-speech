from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.utils.read import load as load_wav
from malaya_speech.utils.tf_featurization import STTFeaturizer
from malaya_speech.utils.subword import load as subword_load
from malaya_speech.model.tf import Transducer, Wav2Vec2_CTC, TransducerAligner
from malaya_speech.path import TRANSDUCER_VOCABS, CTC_VOCABS
import json
import os


def get_vocab(language):
    return TRANSDUCER_VOCABS.get(language, TRANSDUCER_VOCABS['malay'])


def get_vocab_ctc(language):
    return CTC_VOCABS.get(language, CTC_VOCABS['malay'])


dummy_sentences = ['tangan aku disentuh lembut', 'sebut perkataan angka']


def transducer_load(model, module, languages, quantized=False, **kwargs):
    splitted = model.split('-')
    stack = 'stack' == splitted[-2]
    if stack:
        keys = {'model': 'model.pb'}
        for no, language in enumerate(languages):
            keys[f'vocab_{no}'] = get_vocab(language)
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

    time_reduction_factor = {
        'small-conformer': 4,
        'conformer': 4,
        'large-conformer': 4,
        'alconformer': 4,
    }

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
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)
    this_dir = os.path.dirname(__file__)
    wav1, _ = load_wav(os.path.join(this_dir, 'speech', '1.wav'))
    wav2, _ = load_wav(os.path.join(this_dir, 'speech', '2.wav'))

    return Transducer(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        featurizer=featurizer,
        vocab=vocab,
        time_reduction_factor=time_reduction_factor.get(model, 4),
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
        wavs=[wav1, wav2],
        stack=stack,
    )


def transducer_alignment_load(model, module, languages, quantized=False, **kwargs):
    splitted = model.split('-')
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

    time_reduction_factor = {
        'small-conformer': 4,
        'conformer': 4,
        'large-conformer': 4,
        'alconformer': 4,
    }

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
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)
    this_dir = os.path.dirname(__file__)
    wav1, _ = load_wav(os.path.join(this_dir, 'speech', '1.wav'))
    wav2, _ = load_wav(os.path.join(this_dir, 'speech', '2.wav'))

    return TransducerAligner(
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


def wav2vec2_ctc_load(model, module, quantized=False, mode='char', **kwargs):
    if mode == 'char':
        load_vocab = get_vocab_ctc
    else:
        load_vocab = get_vocab

    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.pb',
            'vocab': load_vocab(model.split('-')[-1]),
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    if mode == 'char':
        with open(path['vocab']) as fopen:
            vocab = json.load(fopen) + ['{', '}', '[']
    else:
        vocab = subword_load(path['vocab'].replace('.subwords', ''))

    inputs = ['X_placeholder', 'X_len_placeholder']
    outputs = ['logits', 'seq_lens']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return Wav2Vec2_CTC(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        vocab=vocab,
        sess=generate_session(graph=g, **kwargs),
        mode=mode,
        model=model,
        name=module,
    )
