from malaya_speech.torch_model.synthesis import (
    VITS as VITS_Torch,
    VITS_V2 as VITS_V2_Torch,
)
from malaya_boilerplate.huggingface import download_files
import numpy as np


def vits_torch_load(model, normalizer, v2=False, **kwargs):
    s3_file = {
        'model': 'model.pth',
        'config': 'config.json'
    }
    files = download_files(model, s3_file, **kwargs)
    if v2:
        model = VITS_V2_Torch
    else:
        model = VITS_Torch
    return model(
        normalizer=normalizer,
        pth=files['model'],
        config=files['config'],
        model=model,
        name='text-to-speech-vits',
    )
