from malaya_speech.torch_model.super_resolution import (
    VoiceFixer,
    NVSR,
    NuWave2,
    HiFiGAN,
)
from malaya_boilerplate.huggingface import download_files

repositories = {
    'voicefixer': 'huseinzol05/VoiceFixer',
    'nvsr': 'huseinzol05/NVSR',
    'nuwave2': 'huseinzol05/nuwave2',
    'hifigan-bwe': 'huseinzol05/hifigan-bwe'
}

model_classes = {
    'voicefixer': VoiceFixer,
    'nvsr': NVSR,
    'nuwave2': NuWave2,
    'hifigan-bwe': HiFiGAN,
}

s3_file = {'model': 'model.pth'}


def load_tfgan(model):

    files = download_files(repositories[model], s3_file)
    files_vocoder = download_files('huseinzol05/TFGAN', s3_file)

    return model_classes[model](
        files['model'],
        files_vocoder['model'],
        model=model,
        name='super-resolution',
    )


def load_vocoder(model):
    files = download_files(repositories[model], s3_file)
    return model_classes[model](
        files['model'],
        model=model,
        name='super-resolution',
    )


def load_diffusion(model):
    files = download_files(repositories[model], s3_file)
    return model_classes[model](
        files['model'],
        model=model,
        name='super-resolution',
    )
