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
    'female-singlish': {
        'model': home + '/tts/tacotron2-female-singlish/model.pb',
        'quantized': home + '/tts/tacotron2-female-singlish/quantized/model.pb',
        'stats': home + '/tts/stats/female-singlish.npy',
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
    'female-singlish': {
        'model': 'v2/tts/tacotron2-female-singlish.pb',
        'quantized': 'v2/tts/tacotron2-female-singlish.pb.quantized',
        'stats': 'v1/vocoder-stats/female-singlish.npy',
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
    'female-singlish': {
        'model': home + '/tts/fastspeech2-female-singlish/model.pb',
        'quantized': home
        + '/tts/fastspeech2-female-singlish/quantized/model.pb',
        'stats': home + '/tts/stats/female-singlish.npy',
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
    'female-singlish': {
        'model': 'v2/tts/fastspeech2-female-singlish.pb',
        'quantized': 'v2/tts/fastspeech2-female-singlish.pb.quantized',
        'stats': 'v1/vocoder-stats/female-singlish.npy',
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
