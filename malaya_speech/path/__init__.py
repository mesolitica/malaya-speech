from malaya_boilerplate.utils import _get_home
from malaya_speech import package

home, _ = _get_home(package=package)

TRANSDUCER_VOCAB = 'vocab/transducer.subword.subwords'
TRANSDUCER_MIXED_VOCAB = 'vocab/transducer-mixed-v2.subword.subwords'
TRANSDUCER_SINGLISH_VOCAB = 'vocab/transducer-singlish.subword.subwords'
TRANSDUCER_2048_VOCAB = 'vocab/transducer-2048.subword.subwords'

CTC_VOCAB = 'vocab/ctc-bahasa.json'

TRANSDUCER_VOCABS = {
    'malay': TRANSDUCER_VOCAB,
    'mixed': TRANSDUCER_MIXED_VOCAB,
    'singlish': TRANSDUCER_SINGLISH_VOCAB,
}

CTC_VOCABS = {'malay': CTC_VOCAB}

PATH_TTS_TACOTRON2 = {
    'female': {
        'model': home + '/tts/tacotron2-female/model.pb',
        'quantized': home + '/tts/tacotron2-female/quantized/model.pb',
        'stats': home + '/tts/stats/female.npy',
        'version': 'v2',
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
        'version': 'v2',
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
        'model': 'v2/tts/tacotron2-female.pb',
        'quantized': 'v2/tts/tacotron2-female.pb.quantized',
        'stats': 'v2/vocoder-stats/female.npy',
    },
    'male': {
        'model': 'v2/tts/tacotron2-male.pb',
        'quantized': 'v2/tts/tacotron2-male.pb.quantized',
        'stats': 'v2/vocoder-stats/male.npy',
    },
    'husein': {
        'model': 'v2/tts/tacotron2-husein.pb',
        'quantized': 'v2/tts/tacotron2-husein.pb.quantized',
        'stats': 'v2/vocoder-stats/husein.npy',
    },
    'haqkiem': {
        'model': 'v2/tts/tacotron2-haqkiem.pb',
        'quantized': 'v2/tts/tacotron2-haqkiem.pb.quantized',
        'stats': 'v1/vocoder-stats/haqkiem.npy',
    },
    'female-singlish': {
        'model': 'v3/tts/tacotron2-female-singlish.pb',
        'quantized': 'v3/tts/tacotron2-female-singlish.pb.quantized',
        'stats': 'v3/vocoder-stats/female-singlish.npy',
    },
}

PATH_TTS_FASTSPEECH2 = {
    'female': {
        'model': home + '/tts/fastspeech2-female/model.pb',
        'quantized': home + '/tts/fastspeech2-female/quantized/model.pb',
        'stats': home + '/tts/stats/female.npy',
        'version': 'v2',
    },
    'male': {
        'model': home + '/tts/fastspeech2-male/model.pb',
        'quantized': home + '/tts/fastspeech2-male/quantized/model.pb',
        'stats': home + '/tts/stats/male.npy',
        'version': 'v2',
    },
    'husein': {
        'model': home + '/tts/fastspeech2-husein/model.pb',
        'quantized': home + '/tts/fastspeech2-husein/quantized/model.pb',
        'stats': home + '/tts/stats/husein.npy',
        'version': 'v2',
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
        'version': 'v3',
    },
}

S3_PATH_TTS_FASTSPEECH2 = {
    'female': {
        'model': 'v2/tts/fastspeech2-female.pb',
        'quantized': 'v2/tts/fastspeech2-female.pb.quantized',
        'stats': 'v2/vocoder-stats/female.npy',
    },
    'male': {
        'model': 'v2/tts/fastspeech2-male.pb',
        'quantized': 'v2/tts/fastspeech2-male.pb.quantized',
        'stats': 'v2/vocoder-stats/male.npy',
    },
    'husein': {
        'model': 'v2/tts/fastspeech2-husein.pb',
        'quantized': 'v2/tts/fastspeech2-husein.pb.quantized',
        'stats': 'v2/vocoder-stats/husein.npy',
    },
    'haqkiem': {
        'model': 'v2/tts/fastspeech2-haqkiem.pb',
        'quantized': 'v2/tts/fastspeech2-haqkiem.pb.quantized',
        'stats': 'v1/vocoder-stats/haqkiem.npy',
    },
    'female-singlish': {
        'model': 'v3/tts/fastspeech2-female-singlish.pb',
        'quantized': 'v3/tts/fastspeech2-female-singlish.pb.quantized',
        'stats': 'v3/vocoder-stats/female-singlish.npy',
    },
}

PATH_TTS_FASTPITCH = {
    'female': {
        'model': home + '/tts/fastpitch-female/model.pb',
        'quantized': home + '/tts/fastpitch-female/quantized/model.pb',
        'stats': home + '/tts/stats/female.npy',
        'version': 'v1',
    },
    'male': {
        'model': home + '/tts/fastpitch-male/model.pb',
        'quantized': home + '/tts/fastpitch-male/quantized/model.pb',
        'stats': home + '/tts/stats/male.npy',
        'version': 'v1',
    },
    'husein': {
        'model': home + '/tts/fastpitch-husein/model.pb',
        'quantized': home + '/tts/fastpitch-husein/quantized/model.pb',
        'stats': home + '/tts/stats/husein.npy',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/tts/fastpitch-haqkiem/model.pb',
        'quantized': home + '/tts/fastpitch-haqkiem/quantized/model.pb',
        'stats': home + '/tts/stats/haqkiem.npy',
        'version': 'v1',
    },
    'female-singlish': {
        'model': home + '/tts/fastpitch-female-singlish/model.pb',
        'quantized': home
        + '/tts/fastpitch-female-singlish/quantized/model.pb',
        'stats': home + '/tts/stats/female-singlish.npy',
        'version': 'v1',
    },
}

S3_PATH_TTS_FASTPITCH = {
    'female': {
        'model': 'v1/tts/fastpitch-female.pb',
        'quantized': 'v1/tts/fastpitch-female.pb.quantized',
        'stats': 'v2/vocoder-stats/female.npy',
    },
    'male': {
        'model': 'v1/tts/fastpitch-male.pb',
        'quantized': 'v1/tts/fastpitch-male.pb.quantized',
        'stats': 'v2/vocoder-stats/male.npy',
    },
    'husein': {
        'model': 'v1/tts/fastpitch-husein.pb',
        'quantized': 'v1/tts/fastpitch-husein.pb.quantized',
        'stats': 'v2/vocoder-stats/husein.npy',
    },
    'haqkiem': {
        'model': 'v1/tts/fastpitch-haqkiem.pb',
        'quantized': 'v1/tts/fastpitch-haqkiem.pb.quantized',
        'stats': 'v1/vocoder-stats/haqkiem.npy',
    },
    'female-singlish': {
        'model': 'v1/tts/fastpitch-female-singlish.pb',
        'quantized': 'v1/tts/fastpitch-female-singlish.pb.quantized',
        'stats': 'v1/vocoder-stats/female-singlish.npy',
    },
}
