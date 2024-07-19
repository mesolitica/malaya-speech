from malaya_boilerplate import huggingface
from malaya_speech import package, url


def check_file(file, s3_file=None, **kwargs):
    return huggingface.check_file(file, package, url, s3_file=s3_file, **kwargs)


from . import arange
from . import aligner
from . import astype
from . import char
from . import combine
from . import constant
from . import dist
from . import featurization
from . import generator
from . import griffin_lim
from . import group
from . import io
from . import metrics
from . import nemo_featurization
from . import outlier
from . import padding
from . import read
from . import speechsplit
from . import split
from . import text
from . import subword
from . import text
from . import torch_featurization
from . import validator
