def calculate_cer(actual, hyp):
    """
    Calculate CER using `python-Levenshtein`.
    """
    import Levenshtein as Lev

    actual = actual.replace(' ', '')
    hyp = hyp.replace(' ', '')
    return Lev.distance(actual, hyp) / len(actual)


def calculate_wer(actual, hyp):
    """
    Calculate WER using `python-Levenshtein`.
    """
    import Levenshtein as Lev

    b = set(actual.split() + hyp.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in actual.split()]
    w2 = [chr(word2char[w]) for w in hyp.split()]

    return Lev.distance(''.join(w1), ''.join(w2)) / len(actual.split())
