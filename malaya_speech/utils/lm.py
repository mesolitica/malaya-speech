import collections
import re
import heapq

BeamHypothesis_LM = collections.namedtuple(
    'BeamHypothesis_LM', ('score', 'score_lm', 'prediction', 'states', 'text', 'next_word', 'word_part')
)

BeamHypothesis = collections.namedtuple(
    'BeamHypothesis', ('score', 'prediction', 'states')
)


def prune_space(string):
    return re.sub(r'[ ]+', ' ', string).strip()


def sort_and_trim_beams(beams: list, beam_width: int):
    """
    https://github.com/kensho-technologies/pyctcdecode/blob/v0.1.0/pyctcdecode/decoder.py#L68
    """
    return heapq.nlargest(beam_width, beams, key=lambda x: x.score_lm)


def merge_tokens(token_1, token_2):
    """
    https://github.com/kensho-technologies/pyctcdecode/blob/v0.1.0/pyctcdecode/decoder.py#L100
    """
    if len(token_2) == 0:
        text = token_1
    elif len(token_1) == 0:
        text = token_2
    else:
        text = token_1 + " " + token_2
    return prune_space(text)


def prune_history(beams, lm_order: int):
    """
    https://github.com/kensho-technologies/pyctcdecode/blob/v0.1.0/pyctcdecode/decoder.py#L140

    Filter out beams that are the same over max_ngram history.
    Since n-gram language models have a finite history when scoring a new token, we can use that
    fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
    higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
    some amount of beam diversity. If more than the top beam is used in the output it should
    potentially be disabled.
    """
    # let's keep at least 1 word of history
    min_n_history = max(1, lm_order - 1)
    seen_hashes = set()
    filtered_beams = []
    # for each beam after this, check if we need to add it
    for beam in beams:
        text = beam.text
        next_word = beam.next_word
        word_part = beam.word_part
        last_char = beam.prediction[-1]
        logit_score = beam.score
        hash_idx = (tuple(text.split()[-min_n_history:]), word_part, last_char)
        if hash_idx not in seen_hashes:
            beam_hyp = BeamHypothesis_LM(
                score=logit_score,
                score_lm=0.0,
                prediction=beam.prediction,
                states=beam.states,
                text=text,
                next_word=next_word,
                word_part=word_part
            )
            filtered_beams.append(beam_hyp)
            seen_hashes.add(hash_idx)
    return filtered_beams


def get_lm_beams(beams, cached_lm_scores,
                 cached_partial_token_scores,
                 language_model,
                 is_eos: bool = False):

    new_beams = []
    for beam in beams:
        text = beam.text
        next_word = beam.next_word
        word_part = beam.word_part
        last_char = beam.prediction[-1]
        logit_score = beam.score
        new_text = merge_tokens(text, next_word)
        if new_text not in cached_lm_scores:
            _, prev_raw_lm_score, start_state = cached_lm_scores[text]
            score, end_state = language_model.score(start_state, next_word, is_last_word=is_eos)
            raw_lm_score = prev_raw_lm_score + score
            lm_hw_score = raw_lm_score
            cached_lm_scores[new_text] = (lm_hw_score, raw_lm_score, end_state)
        lm_score, _, _ = cached_lm_scores[new_text]
        if len(word_part) > 0:
            if word_part not in cached_partial_token_scores:
                cached_partial_token_scores[word_part] = language_model.score_partial_token(
                    word_part
                )
            lm_score += cached_partial_token_scores[word_part]
        beam_hyp = BeamHypothesis_LM(
            score=logit_score,
            score_lm=logit_score + lm_score,
            prediction=beam.prediction,
            states=beam.states,
            text=new_text,
            next_word='',
            word_part=word_part
        )
        new_beams.append(beam_hyp)

    return new_beams
