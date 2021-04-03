from malaya_speech import home

TRANSDUCER_VOCAB = 'vocab/transducer.subword.subwords'
TRANSDUCER_MIXED_VOCAB = 'vocab/transducer-mixed.subword.subwords'
TRANSDUCER_SINGLISH_VOCAB = 'vocab/transducer-singlish.subword.subwords'
TRANSDUCER_2048_VOCAB = 'vocab/transducer-2048.subword.subwords'

TRANSDUCER_VOCABS = {
    'malay': TRANSDUCER_VOCAB,
    'mixed': TRANSDUCER_MIXED_VOCAB,
    'singlish': TRANSDUCER_SINGLISH_VOCAB,
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
        'stats': home + '/tts/stats/female.npy',
        'version': 'v1',
    },
    'male': {
        'model': home + '/tts/tacotron2-male/model.pb',
        'quantized': home + '/tts/tacotron2-male/quantized/model.pb',
        'stats': home + '/tts/stats/male.npy',
        'version': 'v1',
    },
    'husein': {
        'model': home + '/tts/tacotron2-husein/model.pb',
        'quantized': home + '/tts/tacotron2-husein/quantized/model.pb',
        'stats': home + '/tts/stats/husein.npy',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/tts/tacotron2-haqkiem/model.pb',
        'quantized': home + '/tts/tacotron2-haqkiem/quantized/model.pb',
        'stats': home + '/tts/stats/haqkiem.npy',
        'version': 'v2',
    },
}

S3_PATH_TTS_TACOTRON2 = {
    'female': {
        'model': 'v1/tts/tacotron2-female.pb',
        'quantized': 'v1/tts/tacotron2-female.pb.quantized',
        'stats': 'v1/vocoder-stats/female.npy',
    },
    'male': {
        'model': 'v1/tts/tacotron2-male.pb',
        'quantized': 'v1/tts/tacotron2-male.pb.quantized',
        'stats': 'v1/vocoder-stats/male.npy',
    },
    'husein': {
        'model': 'v1/tts/tacotron2-husein.pb',
        'quantized': 'v1/tts/tacotron2-husein.pb.quantized',
        'stats': 'v1/vocoder-stats/husein.npy',
    },
    'haqkiem': {
        'model': 'v2/tts/tacotron2-haqkiem.pb',
        'quantized': 'v2/tts/tacotron2-haqkiem.pb.quantized',
        'stats': 'v1/vocoder-stats/haqkiem.npy',
    },
}

PATH_TTS_FASTSPEECH2 = {
    'female': {
        'model': home + '/tts/fastspeech2-female/model.pb',
        'quantized': home + '/tts/fastspeech2-female/quantized/model.pb',
        'stats': home + '/tts/stats/female.npy',
        'version': 'v1',
    },
    'male': {
        'model': home + '/tts/fastspeech2-male/model.pb',
        'quantized': home + '/tts/fastspeech2-male/quantized/model.pb',
        'stats': home + '/tts/stats/male.npy',
        'version': 'v1',
    },
    'husein': {
        'model': home + '/tts/fastspeech2-husein/model.pb',
        'quantized': home + '/tts/fastspeech2-husein/quantized/model.pb',
        'stats': home + '/tts/stats/husein.npy',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/tts/fastspeech2-haqkiem/model.pb',
        'quantized': home + '/tts/fastspeech2-haqkiem/quantized/model.pb',
        'stats': home + '/tts/stats/haqkiem.npy',
        'version': 'v2',
    },
    'female-v2': {
        'model': home + '/tts/fastspeech2-female-v2/model.pb',
        'quantized': home + '/tts/fastspeech2-female-v2/quantized/model.pb',
        'stats': home + '/tts/stats/female.npy',
        'version': 'v1',
    },
    'male-v2': {
        'model': home + '/tts/fastspeech2-male-v2/model.pb',
        'quantized': home + '/tts/fastspeech2-male-v2/quantized/model.pb',
        'stats': home + '/tts/stats/male.npy',
        'version': 'v1',
    },
    'husein-v2': {
        'model': home + '/tts/fastspeech2-husein-v2/model.pb',
        'quantized': home + '/tts/fastspeech2-husein-v2/quantized/model.pb',
        'stats': home + '/tts/stats/husein.npy',
        'version': 'v1',
    },
}

S3_PATH_TTS_FASTSPEECH2 = {
    'female': {
        'model': 'v1/tts/fastspeech2-female.pb',
        'quantized': 'v1/tts/fastspeech2-female.pb.quantized',
        'stats': 'v1/vocoder-stats/female.npy',
    },
    'male': {
        'model': 'v1/tts/fastspeech2-male.pb',
        'quantized': 'v1/tts/fastspeech2-male.pb.quantized',
        'stats': 'v1/vocoder-stats/male.npy',
    },
    'husein': {
        'model': 'v1/tts/fastspeech2-husein.pb',
        'quantized': 'v1/tts/fastspeech2-husein.pb.quantized',
        'stats': 'v1/vocoder-stats/husein.npy',
    },
    'haqkiem': {
        'model': 'v2/tts/fastspeech2-haqkiem.pb',
        'quantized': 'v2/tts/fastspeech2-haqkiem.pb.quantized',
        'stats': 'v1/vocoder-stats/haqkiem.npy',
    },
    'female-v2': {
        'model': 'v1/tts/fastspeech2-female-v2.pb',
        'quantized': 'v1/tts/fastspeech2-female-v2.pb.quantized',
        'stats': 'v1/vocoder-stats/female.npy',
    },
    'male-v2': {
        'model': 'v1/tts/fastspeech2-male-v2.pb',
        'quantized': 'v1/tts/fastspeech2-male-v2.pb.quantized',
        'stats': 'v1/vocoder-stats/male.npy',
    },
    'husein-v2': {
        'model': 'v1/tts/fastspeech2-husein-v2.pb',
        'quantized': 'v1/tts/fastspeech2-husein-v2.pb.quantized',
        'stats': 'v1/vocoder-stats/husein.npy',
    },
}

PATH_VOICE_CONVERSION = {
    'fastvc-32': {
        'model': home + '/vc/fastvc-32/model.pb',
        'quantized': home + '/vc/fastvc-32/quantized/model.pb',
        'version': 'v1',
    },
    'fastvc-64': {
        'model': home + '/vc/fastvc-64/model.pb',
        'quantized': home + '/vc/fastvc-64/quantized/model.pb',
        'version': 'v1',
    },
}

S3_PATH_VOICE_CONVERSION = {
    'fastvc-32': {
        'model': 'v1/vc/fastvc-32.pb',
        'quantized': 'v1/vc/fastvc-32.pb.quantized',
    },
    'fastvc-64': {
        'model': 'v1/vc/fastvc-64.pb',
        'quantized': 'v1/vc/fastvc-64.pb.quantized',
    },
}
