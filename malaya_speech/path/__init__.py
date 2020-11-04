from malaya_speech import home

PATH_AGE_DETECTION = {
    'vggvox-v2': {
        'model': home + '/age-detection/vggvox-v2/model.pb',
        'quantized': home + '/age-detection/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/age-detection/deep-speaker/model.pb',
        'quantized': home + '/age-detection/deep-speaker/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_AGE_DETECTION = {
    'vggvox-v2': {
        'model': 'v1/age-detection/finetuned-vggvox-v2.pb',
        'quantized': 'v1/age-detection/finetuned-vggvox-v2.pb.quantized',
    },
    'deep-speaker': {
        'model': 'v1/age-detection/finetuned-deep-speaker.pb',
        'quantized': 'v1/age-detection/finetuned-deep-speaker.pb.quantized',
    },
}

PATH_EMOTION = {
    'vggvox-v1': {
        'model': home + '/emotion/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/emotion/vggvox-v2/model.pb',
        'quantized': home + '/emotion/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/emotion/deep-speaker/model.pb',
        'quantized': home + '/emotion/deep-speaker/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_EMOTION = {
    'vggvox-v1': {'model': 'v1/emotion/finetuned-vggvox-v1.pb'},
    'vggvox-v2': {
        'model': 'v1/emotion/finetuned-vggvox-v2.pb',
        'quantized': 'v1/emotion/finetuned-vggvox-v2.pb.quantized',
    },
    'deep-speaker': {
        'model': 'v1/emotion/finetuned-deep-speaker.pb',
        'quantized': 'v1/emotion/finetuned-deep-speaker.pb.quantized',
    },
}

PATH_GENDER = {
    'vggvox-v1': {
        'model': home + '/gender/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/gender/vggvox-v2/model.pb',
        'quantized': home + '/gender/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/gender/deep-speaker/model.pb',
        'quantized': home + '/gender/deep-speaker/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_GENDER = {
    'vggvox-v1': {'model': 'v1/gender/finetuned-vggvox-v1.pb'},
    'vggvox-v2': {
        'model': 'v1/gender/finetuned-vggvox-v2.pb',
        'quantized': 'v1/gender/finetuned-vggvox-v2.pb.quantized',
    },
    'deep-speaker': {
        'model': 'v1/gender/finetuned-deep-speaker.pb',
        'quantized': 'v1/gender/finetuned-deep-speaker.pb.quantized',
    },
}

PATH_LANGUAGE_DETECTION = {
    'vggvox-v1': {
        'model': home + '/language-detection/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/language-detection/vggvox-v2/model.pb',
        'quantized': home + '/language-detection/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/language-detection/deep-speaker/model.pb',
        'quantized': home
        + '/language-detection/deep-speaker/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_LANGUAGE_DETECTION = {
    'vggvox-v1': {'model': 'v1/language-detection/finetuned-vggvox-v1.pb'},
    'vggvox-v2': {
        'model': 'v1/language-detection/finetuned-vggvox-v2.pb',
        'quantized': 'v1/language-detection/finetuned-vggvox-v2.pb.quantized',
    },
    'deep-speaker': {
        'model': 'v1/language-detection/finetuned-deep-speaker.pb',
        'quantized': 'v1/language-detection/finetuned-deep-speaker.pb.quantized',
    },
}

PATH_NOISE_REDUCTION = {
    'unet': {
        'model': home + '/noise-reduction/unet.pb',
        'quantized': home + '/noise-reduction/quantized/unet.pb',
        'version': 'v1',
    },
    'resnet-unet': {
        'model': home + '/noise-reduction/resnet-unet.pb',
        'quantized': home + '/noise-reduction/quantized/resnet-unet.pb',
        'version': 'v1',
    },
}

S3_PATH_NOISE_REDUCTION = {
    'unet': {
        'model': 'v1/noise-reduction/unet.pb',
        'quantized': 'v1/noise-reduction/unet.pb.quantized',
    },
    'resnet-unet': {
        'model': 'v1/noise-reduction/resnet-unet.pb',
        'quantized': 'v1/noise-reduction/resnet-unet.pb.quantized',
    },
}

PATH_SPEAKER_CHANGE = {
    'vggvox-v2': {
        'model': home + '/speaker-change/vggvox-v2/model.pb',
        'quantized': home + '/speaker-change/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'speakernet': {
        'model': home + '/speaker-change/speakernet/model.pb',
        'quantized': home + '/speaker-change/speakernet/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_SPEAKER_CHANGE = {
    'vggvox-v2': {
        'model': 'v1/speaker-change/finetuned-vggvox-v2.pb',
        'quantized': 'v1/speaker-change/finetuned-vggvox-v2.pb.quantized',
    },
    'speakernet': {
        'model': 'v1/speaker-change/finetuned-speakernet.pb',
        'quantized': 'v1/speaker-change/finetuned-vggvox-v2.pb.quantized',
    },
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

PATH_SPEAKER_OVERLAP = {
    'vggvox-v2': {
        'model': home + '/speaker-overlap/vggvox-v2/model.pb',
        'quantized': home + '/speaker-overlap/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'speakernet': {
        'model': home + '/speaker-overlap/speakernet/model.pb',
        'quantized': home + '/speaker-overlap/speakernet/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_SPEAKER_OVERLAP = {
    'vggvox-v2': {
        'model': 'v1/speaker-overlap/finetuned-vggvox-v2.pb',
        'quantized': 'v1/speaker-overlap/finetuned-vggvox-v2.pb.quantized',
    },
    'speakernet': {
        'model': 'v1/speaker-overlap/finetuned-speakernet.pb',
        'quantized': 'v1/speaker-overlap/finetuned-vggvox-v2.pb.quantized',
    },
}


PATH_SPEECH_ENHANCEMENT = {
    'unet': {
        'model': home + '/speech-enhancement/unet.pb',
        'quantized': home + '/speech-enhancement/quantized/unet.pb',
        'version': 'v1',
    },
    'resnet-unet': {
        'model': home + '/speech-enhancement/resnet-unet.pb',
        'quantized': home + '/speech-enhancement/quantized/resnet-unet.pb',
        'version': 'v1',
    },
}

S3_PATH_SPEECH_ENHANCEMENT = {
    'unet': {
        'model': 'v1/speech-enhancement/unet.pb',
        'quantized': 'v1/speech-enhancement/unet.pb.quantized',
    },
    'resnet-unet': {
        'model': 'v1/speech-enhancement/resnet-unet.pb',
        'quantized': 'v1/speech-enhancement/resnet-unet.pb.quantized',
    },
}

PATH_VAD = {
    'vggvox-v1': {
        'model': home + '/vad/vggvox-v1/model.pb',
        'quantized': home + '/vad/vggvox-v1/quantized/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/vad/vggvox-v2/model.pb',
        'quantized': home + '/vad/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'inception-v4': {
        'model': home + '/vad/inception-v4/model.pb',
        'version': 'v1',
    },
    'speakernet': {
        'model': home + '/vad/speakernet/model.pb',
        'quantized': home + '/vad/speakernet/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_VAD = {
    'vggvox-v1': {
        'model': 'v1/vad/finetuned-vggvox-v1.pb',
        'quantized': 'v1/vad/finetuned-vggvox-v1.pb.quantized',
    },
    'vggvox-v2': {
        'model': 'v1/vad/finetuned-vggvox-v2.pb',
        'quantized': 'v1/vad/finetuned-vggvox-v2.pb.quantized',
    },
    'inception-v4': {'model': 'v1/vad/finetuned-inception-v4.pb'},
    'speakernet': {
        'model': 'v1/vad/finetuned-speakernet.pb',
        'quantized': 'v1/vad/finetuned-speakernet.pb.quantized',
    },
}
