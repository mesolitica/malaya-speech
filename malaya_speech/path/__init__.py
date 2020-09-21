from malaya_speech import home

PATH_SPEAKER_VECTOR = {
    'pretrained-vggvox-v1': {
        'model': home + 'speaker-vector/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'pretrained-vggvox-v2': {
        'model': home + 'speaker-vector/vggvox-v2/model.pb',
        'version': 'v1',
    },
}

S3_PATH_SPEAKER_VECTOR = {
    'pretrained-vggvox-v1': {
        'model': 'v1/speaker-vector/pretrained-vggvox-v1.pb'
    },
    'pretrained-vggvox-v2': {
        'model': 'v1/speaker-vector/pretrained-vggvox-v2.pb'
    },
}