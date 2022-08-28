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

STATS_VOCODER_FEMALE = 'vocoder-stats/female.npy'
STATS_VOCODER_MALE = 'vocoder-stats/male.npy'
STATS_VOCODER_HAQKIEM = 'vocoder-stats/haqkiem.npy'
STATS_VOCODER_FEMALE_SINGLISH = 'vocoder-stats/female-singlish.npy'
STATS_VOCODER_FEMALE_SINGLISH_V1 = 'vocoder-stats/female-singlish-v1.npy'
STATS_VOCODER_HUSEIN = 'vocoder-stats/husein.npy'
STATS_VOCODER_YASMIN = 'vocoder-stats/yasmin.npy'
STATS_VOCODER_OSMAN = 'vocoder-stats/osman.npy'

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

STATS_VOCODER = {
    'female': STATS_VOCODER_FEMALE,
    'male': STATS_VOCODER_MALE,
    'haqkiem': STATS_VOCODER_HAQKIEM,
    'female-singlish': STATS_VOCODER_FEMALE_SINGLISH,
    'female-singlish-v1': STATS_VOCODER_FEMALE_SINGLISH_V1,
    'husein': STATS_VOCODER_HUSEIN,
    'yasmin': STATS_VOCODER_YASMIN,
    'osman': STATS_VOCODER_OSMAN,
    'osman-sdp': STATS_VOCODER_OSMAN,
    'yasmin-sdp': STATS_VOCODER_YASMIN,
}
