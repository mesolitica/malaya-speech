import numpy as np


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    from scipy.stats import betabinom

    x = np.arange(0, phoneme_count)
    mel_text_probs = []
    for i in range(1, mel_count + 1):
        a, b = scaling_factor * i, scaling_factor * (mel_count + 1 - i)
        mel_i_prob = betabinom(phoneme_count, a, b).pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1

    assert opt.sum(0).all()
    assert opt.sum(1).all()

    return opt


def binarize_attention(attn, in_len, out_len):
    b_size = attn.shape[0]
    attn_cpu = attn
    attn_out = np.zeros_like(attn)
    for ind in range(b_size):
        hard_attn = mas(attn_cpu[ind, 0, : out_len[ind], : in_len[ind]])
        attn_out[ind, 0, : out_len[ind], : in_len[ind]] = hard_attn
    return attn_out


def put_comma(alignment, min_threshold: float = 0.5):
    """
    Put comma in alignment from force alignment model.

    Parameters
    -----------
    alignment: List[Dict[text, start, end]]
    min_threshold: float, optional (default=0.5)
        minimum threshold in term of seconds to assume a comma.

    Returns
    --------
    result: List[str]
    """
    r = []
    for no, row in enumerate(alignment):
        if no > 0:
            if alignment[no]['start'] - alignment[no-1]['end'] >= min_threshold:
                r.append(',')

        r.append(row['text'])
    r.append('.')
    return r
