from malaya_speech.torch_model.super_resolution import VoiceFixer, NVSR
from malaya_boilerplate.huggingface import download_files

repositories = {
    'voicefixer': 'huseinzol05/VoiceFixer',
    'nvsr': 'huseinzol05/NVSR',
}

model_classes = {
    'voicefixer': VoiceFixer,
    'nvsr': NVSR,
}


def load_tfgan(model):
    s3_file = {'model': 'model.pth'}
    files = download_files(repositories[model], s3_file)

    files_vocoder = download_files('huseinzol05/TFGAN', s3_file)
    return model_classes[model](
        files['model'],
        files_vocoder['model'],
        model=model,
        name='super-resolution')
