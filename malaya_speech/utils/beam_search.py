import collections
import numpy as np
import tensorflow as tf

BeamHypothesis = collections.namedtuple(
    'BeamHypothesis', ('score', 'prediction', 'states')
)


def transducer(
    enc,
    total,
    initial_states,
    encoded_placeholder,
    predicted_placeholder,
    states_placeholder,
    ytu,
    new_states,
    sess,
    beam_width = 10,
    norm_score = True,
):
    kept_hyps = [
        BeamHypothesis(score = 0.0, prediction = [0], states = initial_states)
    ]
    B = kept_hyps
    for i in range(total):
        A = B
        B = []
        while True:
            y_hat = max(A, key = lambda x: x.score)
            A.remove(y_hat)
            ytu_, new_states_ = sess.run(
                [ytu, new_states],
                feed_dict = {
                    encoded_placeholder: enc[i],
                    predicted_placeholder: y_hat.prediction[-1],
                    states_placeholder: y_hat.states,
                },
            )
            for k in range(ytu_.shape[0]):
                beam_hyp = BeamHypothesis(
                    score = (y_hat.score + float(ytu_[k])),
                    prediction = y_hat.prediction,
                    states = y_hat.states,
                )
                if k == 0:
                    B.append(beam_hyp)
                else:
                    beam_hyp = BeamHypothesis(
                        score = beam_hyp.score,
                        prediction = (beam_hyp.prediction + [int(k)]),
                        states = new_states_,
                    )
                    A.append(beam_hyp)
            if len(B) > beam_width:
                break
    if norm_score:
        kept_hyps = sorted(
            B, key = lambda x: x.score / len(x.prediction), reverse = True
        )[:beam_width]
    else:
        kept_hyps = sorted(B, key = lambda x: x.score, reverse = True)[
            :beam_width
        ]
    return kept_hyps[0].prediction
