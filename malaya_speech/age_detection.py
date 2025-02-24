from malaya_speech.supervised import classification

_nemo_availability = {
    'huseinzol05/nemo-is-clean-speakernet': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet',
        'Size (MB)': 16.2,
    },
    'huseinzol05/nemo-is-clean-titanet_large': {
        'original from': 'https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet',
        'Size (MB)': 88.8,
    },
}

labels = [
    'teens',
    'twenties',
    'thirties',
    'fourties',
    'fifties',
    'sixties',
    'seventies',
    'eighties',
    'nineties',
    'not an age',
]
