import six

# https://github.com/auspicious3000/SpeechSplit/blob/master/hparams.py


class HParams(object):
    def __init__(self, hparam_def = None, model_structure = None, **kwargs):
        for name, value in six.iteritems(kwargs):
            self.add_hparam(name, value)

    def add_hparam(self, name, value):
        setattr(self, name, value)


hparams = HParams(
    # model
    freq = 8,
    dim_neck = 8,
    freq_2 = 8,
    dim_neck_2 = 1,
    freq_3 = 8,
    dim_neck_3 = 32,
    dim_enc = 512,
    dim_enc_2 = 128,
    dim_enc_3 = 256,
    dim_freq = 80,
    dim_spk_emb = 128,
    dim_f0 = 257,
    dim_dec = 512,
    len_raw = 128,
    chs_grp = 16,
    # interp
    min_len_seg = 19,
    max_len_seg = 32,
    min_len_seq = 64,
    max_len_seq = 128,
    max_len_pad = 192,
)
