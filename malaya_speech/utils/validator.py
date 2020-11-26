from malaya_speech.pipeline import Pipeline


def get_sinker(pipeline, p):
    for p in pipeline.downstreams:
        p = get_sinker(p, p)
    return p


def check_pipeline(object, expected, parameter):
    if isinstance(object, Pipeline):
        object = get_sinker(object, object)
    if expected.lower() not in str(object):
        raise ValueError(
            f'`{parameter}` parameter expected a {expected} module or pipeline.'
        )
