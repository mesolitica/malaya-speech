def f0(model: str = 'fastspeech2', quantized=False, **kwargs):
    """
    Load F0 extractor, Mel-to-F0 model.

    Parameters
    ----------
    model : str, optional (default='fastspeech2')
        Model architecture supported. Allowed values:

        * ``'fastspeech2'`` - fastspeech2 F0 extractor.
        * ``'pitch-extractor'`` - From https://github.com/yl4579/PitchExtractor

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.synthesis.Extractor class
    """


def energy(model: str = 'fastspeech2', quantized=False, **kwargs):
    """
    Load energy extractor, Mel-to-F0 model.

    Parameters
    ----------
    model : str, optional (default='fastspeech2')
        Model architecture supported. Allowed values:

        * ``'fastspeech2'`` - fastspeech2 F0 extractor.
        * ``'pitch-extractor'`` - From https://github.com/yl4579/PitchExtractor

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya_speech.model.synthesis.Extractor class
    """
