from malaya_boilerplate.utils import (
    available_device,
    available_gpu,
    close_session,
    describe_availability,
)
from malaya_boilerplate.frozen_graph import (
    nodes_session,
    generate_session,
    get_device,
)
from malaya_boilerplate import backblaze
from malaya_boilerplate import huggingface
from malaya_boilerplate import frozen_graph
from malaya_boilerplate import utils
from malaya_speech import package, url
import os
import warnings

MALAYA_USE_HUGGINGFACE = os.environ.get('MALAYA_USE_HUGGINGFACE', 'true').lower() == 'true'

if not MALAYA_USE_HUGGINGFACE:
    warnings.warn(
        'os environment `MALAYA_USE_HUGGINGFACE=false` is deprecated, BackBlaze backend no longer maintain after 1.4.0',
        DeprecationWarning)


def print_cache(location=None):
    return utils.print_cache(package=package, location=location)


def delete_cache(location):
    return utils.delete_cache(package=package, location=location)


def delete_all_cache():
    return utils.delete_all_cache(package=package)


def check_file(file, s3_file=None, use_huggingface=True, **kwargs):
    if use_huggingface or MALAYA_USE_HUGGINGFACE:
        return huggingface.check_file(file, package, url, s3_file=s3_file, **kwargs)
    else:
        return backblaze.check_file(file, package, url, s3_file=s3_file, **kwargs)


def load_graph(frozen_graph_filename, **kwargs):
    return frozen_graph.load_graph(package, frozen_graph_filename, **kwargs)


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
from . import tf_featurization
from . import torch_featurization
from . import validator
