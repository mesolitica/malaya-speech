from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.tf import (
    Tacotron,
    Fastspeech,
    Fastpitch,
    GlowTTS,
    GlowTTS_MultiSpeaker
)
from malaya_speech import speaker_vector
import numpy as np


def tacotron_load(
    path, s3_path, model, name, normalizer, quantized=False, **kwargs
):
    check_file(path[model], s3_path[model], quantized=quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder', 'Placeholder_1']
    outputs = ['decoder_output', 'post_mel_outputs', 'alignment_histories']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    stats = np.load(path[model]['stats'])

    return Tacotron(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        normalizer=normalizer,
        stats=stats,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=name,
    )


def fastspeech_load(
    path, s3_path, model, name, normalizer, quantized=False, **kwargs
):
    check_file(path[model], s3_path[model], quantized=quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder', 'speed_ratios', 'f0_ratios', 'energy_ratios']
    outputs = ['decoder_output', 'post_mel_outputs']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    stats = np.load(path[model]['stats'])

    return Fastspeech(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        normalizer=normalizer,
        stats=stats,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=name,
    )


def fastpitch_load(
    path, s3_path, model, name, normalizer, quantized=False, **kwargs
):
    check_file(path[model], s3_path[model], quantized=quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    inputs = ['Placeholder', 'speed_ratios', 'pitch_ratios', 'pitch_addition']
    outputs = ['decoder_output', 'post_mel_outputs', 'pitch_outputs']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    stats = np.load(path[model]['stats'])

    return Fastpitch(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        normalizer=normalizer,
        stats=stats,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=name,
    )


def glowtts_load(
    path, s3_path, model, name, normalizer, quantized=False, **kwargs
):
    check_file(path[model], s3_path[model], quantized=quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    inputs = ['input_ids', 'lens', 'temperature', 'length_ratio']
    if model == 'multispeaker':
        inputs = inputs + ['speakers', 'speakers_right']
        speaker_model = speaker_vector.deep_model('vggvox-v2', **kwargs)
        model_class = GlowTTS_MultiSpeaker
        g = load_graph(path[model][model_path], glowtts_multispeaker_graph=True, **kwargs)
    else:
        speaker_model = None
        model_class = GlowTTS
        g = load_graph(path[model][model_path], glowtts_graph=True, **kwargs)
    outputs = ['mel_output', 'alignment_histories']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    if 'stats' in path[model]:
        stats = np.load(path[model]['stats'])
    else:
        stats = None

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        normalizer=normalizer,
        speaker_vector=speaker_model,
        stats=stats,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=name,
    )
