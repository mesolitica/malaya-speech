from glob import glob
import librosa
import IPython.display as ipd
from scipy.io.wavfile import read
from scipy.signal import resample
from scipy import interpolate
import numpy as np
import os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_audio
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import generator_utils
import tensorflow as tf
import tensor2tensor as t2t
from augmentation import freq_mask, time_mask
from audio_encoder import AudioEncoder, normalize, calc_power_spectrogram
from char_encoder import ByteTextEncoderWithEos
from tensor2tensor.utils import registry
from tensor2tensor import problems
import random

wavs = glob('data/output-wav/*.wav')
wavs.extend(glob('../babble/data/output-wav/*.wav'))


class ByteTextEncoderWithEos(text_encoder.ByteTextEncoder):
    """Encodes each byte to an id and appends the EOS token."""

    def encode(self, s):
        return super(ByteTextEncoderWithEos, self).encode(s) + [
            text_encoder.EOS_ID
        ]


class SpeechRecognitionProblem(problem.Problem):
    """Base class for speech recognition problems."""

    def hparams(self, defaults, model_hparams):
        def add_if_absent(p, attr, value):
            if not hasattr(p, attr):
                p.add_hparam(attr, value)

        p = model_hparams
        add_if_absent(p, 'audio_preproc_in_bottom', False)
        add_if_absent(p, 'audio_keep_example_waveforms', True)
        add_if_absent(p, 'audio_sample_rate', sample_rate)
        add_if_absent(p, 'audio_preemphasis', 0.97)
        add_if_absent(p, 'audio_dither', 1.0 / np.iinfo(np.int16).max)
        add_if_absent(p, 'audio_frame_length', 25.0)
        add_if_absent(p, 'audio_frame_step', 10.0)
        add_if_absent(p, 'audio_lower_edge_hertz', 20.0)
        add_if_absent(p, 'audio_upper_edge_hertz', 8000.0)
        add_if_absent(p, 'audio_num_mel_bins', 80)
        add_if_absent(p, 'audio_add_delta_deltas', False)
        add_if_absent(p, 'num_zeropad_frames', 250)

        p = defaults
        p.modality = {
            'inputs': modalities.ModalityType.SPEECH_RECOGNITION,
            'targets': modalities.ModalityType.SYMBOL,
        }
        p.vocab_size = {'inputs': None, 'targets': 256}

    @property
    def is_character_level(self):
        return True

    @property
    def input_space_id(self):
        return problem.SpaceID.AUDIO_SPECTRAL

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_CHR

    def feature_encoders(self, _):
        return {
            'inputs': None,
            'waveforms': AudioEncoder(sample_rate = sample_rate),
            'targets': ByteTextEncoderWithEos(),
        }

    def example_reading_spec(self):
        data_fields = {
            'waveforms': tf.VarLenFeature(tf.float32),
            'targets': tf.VarLenFeature(tf.int64),
        }

        data_items_to_decoders = None

        return data_fields, data_items_to_decoders


from tqdm import tqdm


@registry.register_problem()
class BahasaSpeech(SpeechRecognitionProblem):
    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [
            {'split': problem.DatasetSplit.TRAIN, 'shards': 99},
            {'split': problem.DatasetSplit.EVAL, 'shards': 1},
        ]

    @property
    def already_shuffled(self):
        return False

    @property
    def use_subword_tokenizer(self):
        return False

    def generator(
        self,
        data_dir,
        tmp_dir,
        datasets,
        eos_list = None,
        start_from = 0,
        how_many = 0,
    ):

        encoders = self.feature_encoders(data_dir)
        audio_encoder = encoders['waveforms']
        text_encoder = encoders['targets']

        for file in tqdm(wavs):
            text_file = file.replace('output-wav', 'output-text') + '.txt'
            with open(text_file) as fopen:
                text_data = fopen.read().strip()

            wav_data, duration = audio_encoder.encode(file)

            if duration < 4:
                continue

            yield {
                'waveforms': wav_data,
                'waveform_lens': [len(wav_data)],
                'targets': text_encoder.encode(text_data),
                'raw_transcript': [text_data],
            }

    def generate_data(self, data_dir, tmp_dir, task_id = -1):

        filepath_fns = {
            problem.DatasetSplit.TRAIN: self.training_filepaths,
            problem.DatasetSplit.EVAL: self.dev_filepaths,
            problem.DatasetSplit.TEST: self.test_filepaths,
        }

        split_paths = [
            (
                split['split'],
                filepath_fns[split['split']](
                    data_dir, split['shards'], shuffled = self.already_shuffled
                ),
            )
            for split in self.dataset_splits
        ]
        all_paths = []
        for _, paths in split_paths:
            all_paths.extend(paths)

        generator_utils.generate_files(
            self.generator(data_dir, tmp_dir, problem.DatasetSplit.TRAIN),
            all_paths,
        )

        generator_utils.shuffle_dataset(all_paths)


DATA_DIR = os.path.expanduser('t2t/data')
TMP_DIR = os.path.expanduser('t2t/tmp')
TRAIN_DIR = os.path.expanduser('t2t/train')
EXPORT_DIR = os.path.expanduser('t2t/export')
TRANSLATIONS_DIR = os.path.expanduser('t2t/translation')
EVENT_DIR = os.path.expanduser('t2t/event')
USR_DIR = os.path.expanduser('t2t/user')

tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(TMP_DIR)
tf.gfile.MakeDirs(TRAIN_DIR)
tf.gfile.MakeDirs(EXPORT_DIR)
tf.gfile.MakeDirs(TRANSLATIONS_DIR)
tf.gfile.MakeDirs(EVENT_DIR)
tf.gfile.MakeDirs(USR_DIR)

PROBLEM = 'bahasa_speech'
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR)

Modes = tf.estimator.ModeKeys

sess = tf.InteractiveSession()
problem_dataset = t2t_problem.dataset(Modes.EVAL, 't2t/data')
problem_dataset = problem_dataset.repeat()
eager_iterator = problem_dataset.make_one_shot_iterator()

r = sess.run(eager_iterator.get_next())

encoders = t2t_problem.feature_encoders(None)


def decode(integers):
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[: integers.index(1)]
    return encoders['targets'].decode(np.squeeze(integers))


decode(r['targets'])
