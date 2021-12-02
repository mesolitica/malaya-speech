from malaya_speech.utils import check_file


def load(model, module, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.trie.klm',
        },
        quantized=False,
        **kwargs,
    )
    return path['model']
