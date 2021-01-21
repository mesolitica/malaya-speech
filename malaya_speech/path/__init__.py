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
        'quantized': home + '/speaker-vector/vggvox-v1/quantized/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/speaker-vector/vggvox-v2/model.pb',
        'quantized': home + '/speaker-vector/vggvox-v2/quantized/model.pb',
        'version': 'v1',
    },
    'inception-v4': {
        'model': home + '/speaker-vector/inception-v4/model.pb',
        'quantized': home + '/speaker-vector/inception-v4/quantized/model.pb',
        'version': 'v1',
    },
    'deep-speaker': {
        'model': home + '/speaker-vector/deep-speaker/model.pb',
        'quantized': home + '/speaker-vector/deep-speaker/quantized/model.pb',
        'version': 'v1',
    },
    'speakernet': {
        'model': home + '/speaker-vector/speakernet/model.pb',
        'quantized': home + '/speaker-vector/speakernet/quantized/model.pb',
        'version': 'v1',
    },
}

S3_PATH_SPEAKER_VECTOR = {
    'vggvox-v1': {
        'model': 'v1/speaker-vector/pretrained-vggvox-v1.pb',
        'quantized': 'v1/speaker-vector/pretrained-vggvox-v1.pb.quantized',
    },
    'vggvox-v2': {
        'model': 'v1/speaker-vector/pretrained-vggvox-v2.pb',
        'model': 'v1/speaker-vector/pretrained-vggvox-v2.pb.quantized',
    },
    'inception-v4': {
        'model': 'v1/speaker-vector/pretrained-inception-v4.pb',
        'quantized': 'v1/speaker-vector/pretrained-inception-v4.pb.quantized',
    },
    'deep-speaker': {
        'model': 'v1/speaker-vector/pretrained-deep-speaker.pb',
        'quantized': 'v1/speaker-vector/pretrained-deep-speaker.pb.quantized',
    },
    'speakernet': {
        'model': 'v1/speaker-vector/pretrained-speakernet.pb',
        'quantized': 'v1/speaker-vector/pretrained-speakernet.pb.quantized',
    },
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
    'masking': {
        'unet': {
            'model': home + '/speech-enhancement/unet/model.pb',
            'quantized': home + '/speech-enhancement/unet/quantized/unet.pb',
            'version': 'v1',
        },
        'resnet-unet': {
            'model': home + '/speech-enhancement/resnet-unet/model.pb',
            'quantized': home
            + '/speech-enhancement/resnet-unet/quantized/model.pb',
            'version': 'v1',
        },
    },
    'enhance': {
        'unet-enhance-24': {
            'model': home + '/speech-enhancement/enhance-24/model.pb',
            'quantized': home
            + '/speech-enhancement/enhance-24/quantized/model.pb',
            'version': 'v1',
        },
        'unet-enhance-36': {
            'model': home + '/speech-enhancement/enhance-36/model.pb',
            'quantized': home
            + '/speech-enhancement/enhance-36/quantized/model.pb',
            'version': 'v1',
        },
    },
}

S3_PATH_SPEECH_ENHANCEMENT = {
    'masking': {
        'unet': {
            'model': 'v1/speech-enhancement/unet.pb',
            'quantized': 'v1/speech-enhancement/unet.pb.quantized',
        },
        'resnet-unet': {
            'model': 'v1/speech-enhancement/resnet-unet.pb',
            'quantized': 'v1/speech-enhancement/resnet-unet.pb.quantized',
        },
    },
    'enhance': {
        'unet-enhance-24': {
            'model': 'v1/speech-enhancement/speech-enhancement-24.pb',
            'quantized': 'v1/speech-enhancement/speech-enhancement-24.pb.quantized',
        },
        'unet-enhance-36': {
            'model': 'v1/speech-enhancement/speech-enhancement-36.pb',
            'quantized': 'v1/speech-enhancement/speech-enhancement-36.pb.quantized',
        },
    },
}

PATH_LM = {
    'malaya-speech': {
        'model': home + '/lm/malaya-speech/model.trie.klm',
        'vocab': home + '/lm/vocab.json',
        'version': 'v1',
    },
    'malaya-speech-wikipedia': {
        'model': home + '/lm/malaya-speech-wikipedia/model.trie.klm',
        'vocab': home + '/lm/vocab.json',
        'version': 'v1',
    },
    'local': {
        'model': home + '/lm/local/model.trie.klm',
        'vocab': home + '/lm/vocab.json',
        'version': 'v1',
    },
    'wikipedia': {
        'model': home + '/lm/wikipedia/model.trie.klm',
        'vocab': home + '/lm/vocab.json',
        'version': 'v1',
    },
}

S3_PATH_LM = {
    'malaya-speech': {
        'model': 'v1/lm/malaya-speech.trie.klm',
        'vocab': 'v1/vocab/malaya-speech-sst-vocab.json',
    },
    'malaya-speech-wikipedia': {
        'model': 'v1/lm/malaya-speech-wiki.trie.klm',
        'vocab': 'v1/vocab/malaya-speech-sst-vocab.json',
    },
    'local': {
        'model': 'v1/lm/local.trie.klm',
        'vocab': 'v1/vocab/malaya-speech-sst-vocab.json',
        'version': 'v1',
    },
    'wikipedia': {
        'model': 'v1/lm/wikipedia.trie.klm',
        'vocab': 'v1/vocab/malaya-speech-sst-vocab.json',
    },
}

PATH_STT_CTC = {
    'mini-jasper': {
        'model': home + '/stt/mini-jasper-ctc/model.pb',
        'quantized': home + '/stt/mini-jasper-ctc/quantized/model.pb',
        'vocab': home + '/stt/vocab/ctc/vocab.json',
        'version': 'v1',
    },
    'medium-jasper': {
        'model': home + '/stt/medium-jasper-ctc/model.pb',
        'quantized': home + '/stt/medium-jasper-ctc/quantized/model.pb',
        'vocab': home + '/stt/vocab/ctc/vocab.json',
        'version': 'v1',
    },
    'jasper': {
        'model': home + '/stt/jasper-ctc/model.pb',
        'quantized': home + '/stt/jasper-ctc/quantized/model.pb',
        'vocab': home + '/stt/vocab/ctc/vocab.json',
        'version': 'v1',
    },
}
S3_PATH_STT_CTC = {
    'mini-jasper': {
        'model': 'v1/stt/mini-jasper-ctc.pb',
        'quantized': 'v1/stt/mini-jasper-ctc.pb.quantized',
        'vocab': 'v1/vocab/malaya-speech-sst-vocab.json',
    },
    'medium-jasper': {
        'model': 'v1/stt/medium-jasper-ctc.pb',
        'quantized': 'v1/stt/medium-jasper-ctc.pb.quantized',
        'vocab': 'v1/vocab/malaya-speech-sst-vocab.json',
    },
    'jasper': {
        'model': 'v1/stt/jasper-ctc.pb',
        'quantized': 'v1/stt/jasper-ctc.pb.quantized',
        'vocab': 'v1/vocab/malaya-speech-sst-vocab.json',
    },
}

PATH_STT_TRANSDUCER = {
    'small-conformer': {
        'model': home + '/stt/small-conformer-transducer/model.pb',
        'quantized': home
        + '/stt/small-conformer-transducer/quantized/model.pb',
        'vocab': home + '/stt/vocab/transducer/vocab.tokenizer.subwords',
        'version': 'v1',
    },
    'conformer': {
        'model': home + '/stt/conformer-transducer/model.pb',
        'quantized': home + '/stt/conformer-transducer/quantized/model.pb',
        'vocab': home + '/stt/vocab/transducer/vocab.tokenizer.subwords',
        'version': 'v1',
    },
    'large-conformer': {
        'model': home + '/stt/large-conformer-transducer/model.pb',
        'quantized': home
        + '/stt/large-conformer-transducer/quantized/model.pb',
        'vocab': home + '/stt/vocab/transducer/vocab.tokenizer.subwords',
        'version': 'v1',
    },
    'alconformer': {
        'model': home + '/stt/alconformer-transducer/model.pb',
        'quantized': home + '/stt/alconformer-transducer/quantized/model.pb',
        'vocab': home + '/stt/vocab/transducer/vocab.tokenizer.subwords',
        'version': 'v1',
    },
}
S3_PATH_STT_TRANSDUCER = {
    'small-conformer': {
        'model': 'v1/stt/small-conformer-transducer.pbb',
        'quantized': 'v1/stt/small-conformer-transducer.pb.quantized',
        'vocab': 'v1/vocab/malaya-speech.tokenizer.subwords',
    },
    'conformer': {
        'model': 'v1/stt/conformer-transducer.pb',
        'quantized': 'v1/stt/conformer-transducer.pb.quantized',
        'vocab': 'v1/vocab/malaya-speech.tokenizer.subwords',
    },
    'large-conformer': {
        'model': 'v1/stt/large-conformer-transducer.pb',
        'quantized': 'v1/stt/large-conformer-transducer.pb.quantized',
        'vocab': 'v1/vocab/malaya-speech.tokenizer.subwords',
    },
    'alconformer': {
        'model': 'v1/stt/alconformer-transducer.pb',
        'quantized': 'v1/stt/alconformer-transducer.pb.quantized',
        'vocab': 'v1/vocab/malaya-speech.tokenizer.subwords',
    },
}

PATH_SUPER_RESOLUTION = {
    'srgan-128': {
        'model': home + '/super-resolution/srgan-128/model.pb',
        'quantized': home + '/super-resolution/srgan-128/quantized/model.pb',
        'version': 'v1',
    },
    'srgan-256': {
        'model': home + '/super-resolution/srgan-256/model.pb',
        'quantized': home + '/super-resolution/srgan-256/quantized/model.pb',
        'version': 'v1',
    },
}
S3_PATH_SUPER_RESOLUTION = {
    'srgan-128': {
        'model': 'v1/super-resolution/srgan-128.pb',
        'quantized': 'v1/super-resolution/srgan-128.pb.quantized',
    },
    'srgan-256': {
        'model': 'v1/super-resolution/srgan-256.pb',
        'quantized': 'v1/super-resolution/srgan-256.pb.quantized',
    },
}

PATH_TTS_TACOTRON2 = {
    'female': {
        'model': home + '/tts/tacotron2-female/model.pb',
        'quantized': home + '/tts/tacotron2-female/quantized/model.pb',
        'version': 'v1',
    },
    'male': {
        'model': home + '/tts/tacotron2-male/model.pb',
        'quantized': home + '/tts/tacotron2-male/quantized/model.pb',
        'version': 'v1',
    },
    'husein': {
        'model': home + '/tts/tacotron2-husein/model.pb',
        'quantized': home + '/tts/tacotron2-husein/quantized/model.pb',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/tts/tacotron2-haqkiem/model.pb',
        'quantized': home + '/tts/tacotron2-haqkiem/quantized/model.pb',
        'version': 'v2',
    },
}

S3_PATH_TTS_TACOTRON2 = {
    'female': {
        'model': 'v1/tts/tacotron2-female.pb',
        'quantized': 'v1/tts/tacotron2-female.pb.quantized',
    },
    'male': {
        'model': 'v1/tts/tacotron2-male.pb',
        'quantized': 'v1/tts/tacotron2-male.pb.quantized',
    },
    'husein': {
        'model': 'v1/tts/tacotron2-husein.pb',
        'quantized': 'v1/tts/tacotron2-husein.pb.quantized',
    },
    'haqkiem': {
        'model': 'v2/tts/tacotron2-haqkiem.pb',
        'quantized': 'v2/tts/tacotron2-haqkiem.pb.quantized',
    },
}

PATH_TTS_FASTSPEECH2 = {
    'female': {
        'model': home + '/tts/fastspeech2-female/model.pb',
        'quantized': home + '/tts/fastspeech2-female/quantized/model.pb',
        'version': 'v1',
    },
    'male': {
        'model': home + '/tts/fastspeech2-male/model.pb',
        'quantized': home + '/tts/fastspeech2-male/quantized/model.pb',
        'version': 'v1',
    },
    'husein': {
        'model': home + '/tts/fastspeech2-husein/model.pb',
        'quantized': home + '/tts/fastspeech2-husein/quantized/model.pb',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/tts/fastspeech2-haqkiem/model.pb',
        'quantized': home + '/tts/fastspeech2-haqkiem/quantized/model.pb',
        'version': 'v2',
    },
    'female-v2': {
        'model': home + '/tts/fastspeech2-female-v2/model.pb',
        'quantized': home + '/tts/fastspeech2-female-v2/quantized/model.pb',
        'version': 'v1',
    },
    'male-v2': {
        'model': home + '/tts/fastspeech2-male-v2/model.pb',
        'quantized': home + '/tts/fastspeech2-male-v2/quantized/model.pb',
        'version': 'v1',
    },
    'husein-v2': {
        'model': home + '/tts/fastspeech2-husein-v2/model.pb',
        'quantized': home + '/tts/fastspeech2-husein-v2/quantized/model.pb',
        'version': 'v1',
    },
}

S3_PATH_TTS_FASTSPEECH2 = {
    'female': {
        'model': 'v1/tts/fastspeech2-female.pb',
        'quantized': 'v1/tts/fastspeech2-female.pb.quantized',
    },
    'male': {
        'model': 'v1/tts/fastspeech2-male.pb',
        'quantized': 'v1/tts/fastspeech2-male.pb.quantized',
    },
    'husein': {
        'model': 'v1/tts/fastspeech2-husein.pb',
        'quantized': 'v1/tts/fastspeech2-husein.pb.quantized',
    },
    'haqkiem': {
        'model': 'v2/tts/fastspeech2-haqkiem.pb',
        'quantized': 'v2/tts/fastspeech2-haqkiem.pb.quantized',
    },
    'female-v2': {
        'model': 'v1/tts/fastspeech2-female-v2.pb',
        'quantized': 'v1/tts/fastspeech2-female-v2.pb.quantized',
    },
    'male-v2': {
        'model': 'v1/tts/fastspeech2-male-v2.pb',
        'quantized': 'v1/tts/fastspeech2-male-v2.pb.quantized',
    },
    'husein-v2': {
        'model': 'v1/tts/fastspeech2-husein-v2.pb',
        'quantized': 'v1/tts/fastspeech2-husein-v2.pb.quantized',
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

PATH_VOCODER_MELGAN = {
    'male': {
        'model': home + '/vocoder/melgan-male/model.pb',
        'quantized': home + '/vocoder/melgan-male/quantized/model.pb',
        'version': 'v1',
    },
    'female': {
        'model': home + '/vocoder/melgan-female/model.pb',
        'quantized': home + '/vocoder/melgan-female/quantized/model.pb',
        'version': 'v1',
    },
    'husein': {
        'model': home + '/vocoder/melgan-husein/model.pb',
        'quantized': home + '/vocoder/melgan-husein/quantized/model.pb',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/vocoder/melgan-haqkiem/model.pb',
        'quantized': home + '/vocoder/melgan-haqkiem/quantized/model.pb',
        'version': 'v2',
    },
}

PATH_VOCODER_MBMELGAN = {
    'male': {
        'model': home + '/vocoder/mbmelgan-male/model.pb',
        'quantized': home + '/vocoder/mbmelgan-male/quantized/model.pb',
        'version': 'v1',
    },
    'female': {
        'model': home + '/vocoder/mbmelgan-female/model.pb',
        'quantized': home + '/vocoder/mbmelgan-female/quantized/model.pb',
        'version': 'v1',
    },
    'husein': {
        'model': home + '/vocoder/mbmelgan-husein/model.pb',
        'quantized': home + '/vocoder/mbmelgan-husein/quantized/model.pb',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/vocoder/mbmelgan-haqkiem/model.pb',
        'quantized': home + '/vocoder/mbmelgan-haqkiem/quantized/model.pb',
        'version': 'v1',
    },
}


S3_PATH_VOCODER_MELGAN = {
    'male': {
        'model': 'v1/vocoder/melgan-male.pb',
        'quantized': 'v1/vocoder/melgan-male.pb.quantized',
    },
    'female': {
        'model': 'v1/vocoder/melgan-female.pb',
        'quantized': 'v1/vocoder/melgan-female.pb.quantized',
    },
    'husein': {
        'model': 'v1/vocoder/melgan-husein.pb',
        'quantized': 'v1/vocoder/melgan-husein.pb.quantized',
    },
    'haqkiem': {
        'model': 'v2/vocoder/melgan-haqkiem.pb',
        'quantized': 'v2/vocoder/melgan-haqkiem.pb.quantized',
    },
}

S3_PATH_VOCODER_MBMELGAN = {
    'male': {
        'model': 'v1/vocoder/mbmelgan-male.pb',
        'quantized': 'v1/vocoder/mbmelgan-male.pb.quantized',
    },
    'female': {
        'model': 'v1/vocoder/mbmelgan-female.pb',
        'quantized': 'v1/vocoder/mbmelgan-female.pb.quantized',
    },
    'husein': {
        'model': 'v1/vocoder/mbmelgan-husein.pb',
        'quantized': 'v1/vocoder/mbmelgan-husein.pb.quantized',
    },
    'haqkiem': {
        'model': 'v2/vocoder/mbmelgan-haqkiem.pb',
        'quantized': 'v2/vocoder/mbmelgan-haqkiem.pb.quantized',
    },
}
