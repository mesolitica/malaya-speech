def check_pipeline(pipeline, expected):
    if hasattr(pipeline, 'pipeline'):
        p = pipeline.pipeline[-1]
    else:
        p = pipeline

    if p.__name__ != expected:
        raise ValueError(f'Expected output of pipeline / model is {expected}')
