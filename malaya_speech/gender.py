from malaya_speech.supervised import classification
from herpetologist import check_type

_availability = {
    'vggvox-v1': {'Size (MB)': 70.8, 'Accuracy': 0},
    'vggvox-v2': {'Size (MB)': 31.1, 'Accuracy': 0.959375},
}


def available_model():
    from malaya_speech.utils import describe_availability

    return describe_availability(_availability)


@check_type
def deep_model(model: str = 'vggvox-v2', **kwargs):
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya_speech.gender.available_model()'
        )
