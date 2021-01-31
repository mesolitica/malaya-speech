from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.model.tf import FastVC
from malaya_speech.utils.featurization import universal_mel
from malaya_speech import speaker_vector


def load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)

    speaker_model = speaker_vector.deep_model('vggvox-v2', **kwargs)
    output_nodes = ['mel_before', 'mel_after']
    outputs = {n: g.get_tensor_by_name(f'import/{n}:0') for n in output_nodes}
    return FastVC(
        mel = g.get_tensor_by_name('import/mel:0'),
        ori_vector = g.get_tensor_by_name('import/ori_vector:0'),
        target_vector = g.get_tensor_by_name('import/target_vector:0'),
        mel_lengths = g.get_tensor_by_name('import/mel_lengths:0'),
        logits = outputs,
        waveform_to_mel = universal_mel,
        speaker_vector = speaker_model,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )
