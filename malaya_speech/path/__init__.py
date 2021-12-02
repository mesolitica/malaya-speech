from malaya_boilerplate.utils import _get_home
from malaya_speech import package

home, _ = _get_home(package=package)

TRANSDUCER_VOCAB = 'vocab/transducer.subword.subwords'
TRANSDUCER_MIXED_VOCAB = 'vocab/transducer-mixed-v2.subword.subwords'
TRANSDUCER_SINGLISH_VOCAB = 'vocab/transducer-singlish.subword.subwords'
TRANSDUCER_2048_VOCAB = 'vocab/transducer-2048.subword.subwords'
TRANSDUCER_BAHASA_512_VOCAB = 'vocab/bahasa-512.subword.subwords'
TRANSDUCER_SINGLISH_512_VOCAB = 'vocab/singlish-512.subword.subwords'
TRANSDUCER_MANDARIN_512_VOCAB = 'vocab/mandarin-512.subword.subwords'

TRANSDUCER_VOCABS = {
    'malay': TRANSDUCER_VOCAB,
    'mixed': TRANSDUCER_MIXED_VOCAB,
    'singlish': TRANSDUCER_SINGLISH_VOCAB,
}

TRANSDUCER_MIXED_VOCABS = {
    'malay': TRANSDUCER_BAHASA_512_VOCAB,
    'singlish': TRANSDUCER_SINGLISH_512_VOCAB,
    'mandarin': TRANSDUCER_MANDARIN_512_VOCAB,
}

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

PATH_TTS_GLOWTTS = {
    'female': {
        'model': home + '/tts/glowtts-female/model.pb',
        'quantized': home + '/tts/glowtts-female/quantized/model.pb',
        'stats': home + '/tts/stats/female.npy',
        'version': 'v1',
    },
    'male': {
        'model': home + '/tts/glowtts-male/model.pb',
        'quantized': home + '/tts/glowtts-male/quantized/model.pb',
        'stats': home + '/tts/stats/male.npy',
        'version': 'v1',
    },
    'haqkiem': {
        'model': home + '/tts/glowtts-haqkiem/model.pb',
        'quantized': home + '/tts/glowtts-haqkiem/quantized/model.pb',
        'stats': home + '/tts/stats/haqkiem.npy',
        'version': 'v1',
    },
    'female-singlish': {
        'model': home + '/tts/glowtts-female-singlish/model.pb',
        'quantized': home + '/tts/glowtts-female-singlish/quantized/model.pb',
        'stats': home + '/tts/stats/female-singlish.npy',
        'version': 'v1',
    },
    'multispeaker': {
        'model': home + '/tts/glowtts-multispeaker/model.pb',
        'quantized': home + '/tts/glowtts-multispeaker/quantized/model.pb',
        'version': 'v1',
    },
}

S3_PATH_TTS_GLOWTTS = {
    'female': {
        'model': 'v2/tts/glowtts-female.pb',
        'quantized': 'v2/tts/glowtts-female.pb.quantized',
        'stats': 'v2/vocoder-stats/female.npy',
    },
    'male': {
        'model': 'v2/tts/glowtts-male.pb',
        'quantized': 'v2/tts/glowtts-male.pb.quantized',
        'stats': 'v2/vocoder-stats/male.npy',
    },
    'haqkiem': {
        'model': 'v2/tts/glowtts-haqkiem.pb',
        'quantized': 'v2/tts/glowtts-haqkiem.pb.quantized',
        'stats': 'v1/vocoder-stats/haqkiem.npy',
    },
    'female-singlish': {
        'model': 'v2/tts/glowtts-female-singlish.pb',
        'quantized': 'v2/tts/glowtts-female-singlish.pb.quantized',
        'stats': 'v1/vocoder-stats/female-singlish.npy',
    },
    'multispeaker': {
        'model': 'v2/tts/glowtts-multispeaker.pb',
        'quantized': 'v2/tts/glowtts-multispeaker.pb.quantized',
    },
}
