from malaya_speech.utils import check_file, load_graph, generate_session
from malaya_speech.model.tf import TACOTRON


def tacotron_load(
    path, s3_path, model, name, normalizer, quantized = False, **kwargs
):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    output_nodes = ['decoder_output', 'post_mel_outputs', 'alignment_histories']
    outputs = {n: g.get_tensor_by_name(f'import/{n}:0') for n in output_nodes}

    return TACOTRON(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        X_len = g.get_tensor_by_name('import/Placeholder_1:0'),
        logits = outputs,
        normalizer = normalizer,
        sess = generate_session(graph = g, **kwargs),
        model = model,
        name = name,
    )


def fastspeech_load(path, s3_path, model, name, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'
