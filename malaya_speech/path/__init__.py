from malaya_speech import home

PATH_SPEAKER_VECTOR = {
    'vggvox-v1': {
        'model': home + '/speaker-vector/vggvox-v1/model.pb',
        'version': 'v1',
    },
    'vggvox-v2': {
        'model': home + '/speaker-vector/vggvox-v2/model.pb',
        'version': 'v1',
    },
}

S3_PATH_SPEAKER_VECTOR = {
    'vggvox-v1': {'model': 'v1/speaker-vector/pretrained-vggvox-v1.pb'},
    'vggvox-v2': {'model': 'v1/speaker-vector/pretrained-vggvox-v2.pb'},
}

PATH_VAD = {
    'vggvox-v2': {'model': home + '/vad/vggvox-v2/model.pb', 'version': 'v1'}
}
S3_PATH_VAD = {'vggvox-v2': {'model': 'v1/vad/finetuned-vggvox-v2.pb'}}
