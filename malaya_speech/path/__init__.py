from malaya_speech import home

PATH_AGE_DETECTION = {
    'vggvox-v2': {
        'model': home + '/age-detection/vggvox-v2/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/age-detection/deep-speaker/model.pb',
        'version': 'v1',
    },
}
S3_PATH_AGE_DETECTION = {
    'vggvox-v2': {'model': 'v1/age-detection/finetuned-vggvox-v2.pb'},
    'deep-speaker': {'model': 'v1/age-detection/finetuned-deep-speaker.pb'},
}

PATH_EMOTION = {
    'vggvox-v1': {
        'model': home + '/emotion/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/emotion/vggvox-v2/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/emotion/deep-speaker/model.pb',
        'version': 'v1',
    },
}
S3_PATH_EMOTION = {
    'vggvox-v1': {'model': 'v1/emotion/finetuned-vggvox-v1.pb'},
    'vggvox-v2': {'model': 'v1/emotion/finetuned-vggvox-v2.pb'},
    'deep-speaker': {'model': 'v1/emotion/finetuned-deep-speaker.pb'},
}

PATH_GENDER = {
    'vggvox-v1': {
        'model': home + '/gender/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/gender/vggvox-v2/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/gender/deep-speaker/model.pb',
        'version': 'v1',
    },
}
S3_PATH_GENDER = {
    'vggvox-v1': {'model': 'v1/gender/finetuned-vggvox-v1.pb'},
    'vggvox-v2': {'model': 'v1/gender/finetuned-vggvox-v2.pb'},
    'deep-speaker': {'model': 'v1/gender/finetuned-deep-speaker.pb'},
}

PATH_LANGUAGE_DETECTION = {
    'vggvox-v1': {
        'model': home + '/language-detection/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/language-detection/vggvox-v2/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/language-detection/deep-speaker/model.pb',
        'version': 'v1',
    },
}
S3_PATH_LANGUAGE_DETECTION = {
    'vggvox-v1': {'model': 'v1/language-detection/finetuned-vggvox-v1.pb'},
    'vggvox-v2': {'model': 'v1/language-detection/finetuned-vggvox-v2.pb'},
    'deep-speaker': {
        'model': 'v1/language-detection/finetuned-deep-speaker.pb'
    },
}

PATH_NOISE_REDUCTION = {
    'unet': {'model': home + '/noise-reduction/unet.pb', 'version': 'v1'},
    'resnet34-unet': {
        'model': home + '/noise-reduction/resnet34-unet.pb',
        'version': 'v1',
    },
}

S3_PATH_NOISE_REDUCTION = {
    'unet': {'model': 'v1/noise-reduction/unet.pb'},
    'resnet34-unet': {'model': 'v1/noise-reduction/resnet34-unet.pb'},
}

PATH_SPEAKER_VECTOR = {
    'vggvox-v1': {
        'model': home + '/speaker-vector/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/speaker-vector/vggvox-v2/model.pb',
        'version': 'v1',
    },
    'inception-v4': {
        'model': home + '/speaker-vector/inception-v4/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/speaker-vector/deep-speaker/model.pb',
        'version': 'v1',
    },
    'speakernet': {
        'model': home + '/speaker-vector/speakernet/model.pb',
        'version': 'v1',
    },
}

S3_PATH_SPEAKER_VECTOR = {
    'vggvox-v1': {'model': 'v1/speaker-vector/pretrained-vggvox-v1.pb'},
    'vggvox-v2': {'model': 'v1/speaker-vector/pretrained-vggvox-v2.pb'},
    'inception-v4': {'model': 'v1/speaker-vector/pretrained-inception-v4.pb'},
    'deep-speaker': {'model': 'v1/speaker-vector/pretrained-deep-speaker.pb'},
    'speakernet': {'model': 'v1/speaker-vector/pretrained-speakernet.pb'},
}

PATH_SPEECH_ENHANCEMENT = {
    'resnet34-unet': {
        'model': home + '/speech-enhancement/resnet34-unet.pb',
        'version': 'v1',
    },
    'inception-v3-unet': {
        'model': home + '/speech-enhancement/inception-v3-unet.pb',
        'version': 'v1',
    },
}

S3_PATH_SPEECH_ENHANCEMENT = {
    'resnet34-unet': {'model': 'v1/speech-enhancement/resnet34-unet.pb'},
    'inception-v3-unet': {
        'model': 'v1/speech-enhancement/inception-v3-unet.pb'
    },
}

PATH_VAD = {
    'vggvox-v1': {'model': home + '/vad/vggvox-v1/model.pb', 'version': 'v1'},
    'vggvox-v2': {'model': home + '/vad/vggvox-v2/model.pb', 'version': 'v1'},
    'inception-v4': {
        'model': home + '/vad/inception-v4/model.pb',
        'version': 'v1',
    },
}
S3_PATH_VAD = {
    'vggvox-v1': {'model': 'v1/vad/finetuned-vggvox-v1.pb'},
    'vggvox-v2': {'model': 'v1/vad/finetuned-vggvox-v2.pb'},
    'inception-v4': {'model': 'v1/vad/finetuned-inception-v4.pb'},
}
