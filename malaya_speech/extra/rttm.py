from herpetologist import check_type


@check_type
def load(file: str):
    """
    Load RTTM file.

    Parameters
    ----------
    file: str

    Returns
    -------
    result : Dict[str, malaya_speech.model.annotation.Annotation]
    """
    from malaya_speech.model.annotation import Annotation
    from malaya_speech.model.frame import Segment

    try:
        import pandas as pd
    except:
        raise ValueError(
            'pandas not installed. Please install it by `pip install pandas` and try again.'
        )

    names = [
        'NA1',
        'uri',
        'NA2',
        'start',
        'duration',
        'NA3',
        'NA4',
        'speaker',
        'NA5',
        'NA6',
    ]
    dtype = {'uri': str, 'start': float, 'duration': float, 'speaker': str}
    data = pd.read_csv(
        file,
        names = names,
        dtype = dtype,
        delim_whitespace = True,
        keep_default_na = False,
    )
    annotations = {}
    for uri, turns in data.groupby('uri'):
        annotation = Annotation(uri)
        for i, turn in turns.iterrows():
            segment = Segment(turn['start'], turn['start'] + turn['duration'])
            annotation[segment, i] = turn['speaker']
        annotations[uri] = annotation
    return annotations
