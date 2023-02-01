from malaya_speech.utils import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya_speech.model.synthesis import (
    Tacotron,
    Fastspeech,
    FastspeechSDP,
    E2E_FastSpeech,
    Fastpitch,
    GlowTTS,
    GlowTTS_MultiSpeaker,
    VITS,
)
from malaya_speech.torch_model.synthesis import VITS as VITS_Torch
from malaya_boilerplate.huggingface import download_files
from malaya_speech import speaker_vector
from malaya_speech.path import STATS_VOCODER
import numpy as np


def load(model, module, inputs, outputs, normalizer, model_class, quantized=False, **kwargs,):
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'stats': STATS_VOCODER[model]},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)
    stats = np.load(path['stats'])
    sess = generate_session(graph=g, **kwargs)
    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        normalizer=normalizer,
        stats=stats,
        sess=sess,
        model=model,
        name=module,
    )


def tacotron_load(
    model, module, normalizer, quantized=False, **kwargs,
):
    inputs = ['Placeholder', 'Placeholder_1']
    outputs = ['decoder_output', 'post_mel_outputs', 'alignment_histories']
    return load(
        model=model,
        module=module,
        inputs=inputs,
        outputs=outputs,
        normalizer=normalizer,
        model_class=Tacotron,
        quantized=quantized,
        **kwargs,
    )


def fastspeech_load(
    model, module, normalizer, quantized=False, **kwargs,
):
    sdp = model.split('-')[-1] == 'sdp'
    inputs = ['Placeholder', 'speed_ratios', 'f0_ratios', 'energy_ratios']
    outputs = ['decoder_output', 'post_mel_outputs']

    if sdp:
        inputs.append('noise_scale_w')
        model_class = FastspeechSDP
    else:
        model_class = Fastspeech

    return load(
        model=model,
        module=module,
        inputs=inputs,
        outputs=outputs,
        normalizer=normalizer,
        model_class=model_class,
        quantized=quantized,
        **kwargs,
    )


def fastpitch_load(
    model, module, normalizer, quantized=False, **kwargs,
):
    inputs = ['Placeholder', 'speed_ratios', 'pitch_ratios', 'pitch_addition']
    outputs = ['decoder_output', 'post_mel_outputs', 'pitch_outputs']
    return load(
        model=model,
        module=module,
        inputs=inputs,
        outputs=outputs,
        normalizer=normalizer,
        model_class=Fastpitch,
        quantized=quantized,
        **kwargs,
    )


def glowtts_load(
    model, module, normalizer, quantized=False, **kwargs,
):
    if model == 'female-singlish':
        stats = f'{model}-v1'
    else:
        stats = model
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'stats': STATS_VOCODER.get(stats, 'male')},
        quantized=quantized,
        **kwargs,
    )

    inputs = ['input_ids', 'lens', 'temperature', 'length_ratio']
    if model == 'multispeaker':
        inputs = inputs + ['speakers', 'speakers_right']
        g = load_graph(path['model'], glowtts_multispeaker_graph=True, **kwargs)
        speaker_model = speaker_vector.deep_model('vggvox-v2', **kwargs)
        model_class = GlowTTS_MultiSpeaker
        stats = None
    else:
        speaker_model = None
        model_class = GlowTTS
        g = load_graph(path['model'], glowtts_graph=True, **kwargs)
        stats = np.load(path['stats'])

    outputs = ['mel_output', 'alignment_histories']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)
    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        normalizer=normalizer,
        speaker_vector=speaker_model,
        stats=stats,
        sess=generate_session(graph=g, **kwargs),
        model=model,
        name=module,
    )


def vits_load(
    model, module, normalizer, quantized=False, **kwargs,
):
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb'},
        quantized=quantized,
        **kwargs,
    )

    inputs = ['input_ids', 'lens', 'temperature', 'length_ratio']
    if 'sdp' in model:
        inputs.append('noise_scale_w')

    outputs = ['mel_output', 'alignment_histories', 'y_hat']
    g = load_graph(path['model'], **kwargs)
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)
    sess = generate_session(graph=g, **kwargs)
    return VITS(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        normalizer=normalizer,
        sess=sess,
        model=model,
        name=module,
    )


def vits_torch_load(model, normalizer, **kwargs):
    s3_file = {
        'model': 'model.pth',
        'config': 'config.json'
    }
    files = download_files(model, s3_file, **kwargs)
    return VITS_Torch(
        normalizer=normalizer,
        pth=files['model'],
        config=files['config'],
        model=model,
        name='text-to-speech-vits',
    )


def e2e_fastspeech_load(
    model, module, normalizer, quantized=False, **kwargs,
):
    inputs = ['Placeholder', 'speed_ratios', 'f0_ratios', 'energy_ratios', 'noise_scale_w']
    outputs = ['y_hat']

    return load(
        model=model,
        module=module,
        inputs=inputs,
        outputs=outputs,
        normalizer=normalizer,
        model_class=E2E_FastSpeech,
        quantized=quantized,
        **kwargs,
    )
