import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import sys
import string
import torch
import numpy as np
import transformers
import requests
import datasets
from torch import nn
from glob import glob
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from packaging import version
import tensorflow as tf
import random
import json
import logging
import shutil
from multiprocessing import Pool


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

CTC_VOCAB = [''] + list(string.ascii_lowercase + string.digits) + [' ']


def download_file_cloud(url, filename):
    r = requests.get(url, stream=True)
    total_size = int(r.headers['content-length'])
    version = int(r.headers.get('X-Bz-Upload-Timestamp', 0))
    try:
        local_size = os.path.getsize(filename)
        if local_size == total_size:
            print(f'{filename} local size matched with cloud size')
            return version
    except Exception as e:
        print(e)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for data in r.iter_content(chunk_size=1_048_576):
            f.write(data)


def get_dataset(files, directory='tfrecord', overwrite_directory=True):
    os.makedirs(directory, exist_ok=True)
    if overwrite_directory:
        shutil.rmtree(directory)
    files_to_download = []
    for f in files:
        filename = os.path.join(directory, '-'.join(f.split('/')[-2:]))
        files_to_download.append((f, filename))

    pool = Pool(processes=len(files))
    pool.starmap(download_file_cloud, files_to_download)
    pool.close()
    pool.join()
    tfrecords = glob(f'{directory}/*.tfrecord')
    return tfrecords


def parse(serialized_example):

    data_fields = {
        'waveforms': tf.compat.v1.VarLenFeature(tf.float32),
        'targets': tf.compat.v1.VarLenFeature(tf.int64),
        'targets_length': tf.compat.v1.VarLenFeature(tf.int64),
        'lang': tf.compat.v1.VarLenFeature(tf.int64),
    }
    features = tf.compat.v1.parse_single_example(
        serialized_example, features=data_fields
    )
    for k in features.keys():
        features[k] = features[k].values

    keys = list(features.keys())
    for k in keys:
        if k not in ['waveforms', 'waveforms_length', 'targets']:
            features.pop(k, None)

    return features


class MalayaDataset(torch.utils.data.Dataset):
    def __init__(self, files, directory, batch_files=10, max_batch=999999, overwrite_directory=True,
                 start=False):
        self.files = [t.replace('gs://mesolitica-tpu-general',
                                'https://huggingface.co/huseinzol05/STT-Mixed-TFRecord/resolve/main') for t in files]
        self.directory = directory
        self.batch_files = batch_files
        self.i = 0
        self.d = None
        self.sr = 16000
        self.maxlen = 16
        self.minlen = 2
        self.max_batch = max_batch
        self.overwrite_directory = overwrite_directory
        if start:
            self.get_dataset()

    def get_dataset(self, num_cpu_threads=4, thread_count=12):
        if self.i >= len(self.files) or self.i == 0:
            self.i = 0
            random.shuffle(self.files)
        b = self.files[self.i: self.i + self.batch_files]
        tfrecords = get_dataset(b, directory=self.directory, overwrite_directory=self.overwrite_directory)
        d = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecords))
        d = d.shuffle(buffer_size=len(tfrecords))
        cycle_length = min(num_cpu_threads, len(tfrecords))
        d = d.interleave(
            tf.data.TFRecordDataset,
            cycle_length=cycle_length,
            block_length=thread_count)
        d = d.shuffle(buffer_size=100)
        d = d.map(parse, num_parallel_calls=num_cpu_threads)
        d = d.filter(
            lambda x: tf.less(tf.shape(x['waveforms'])[0] / self.sr, self.maxlen)
        )
        d = d.filter(
            lambda x: tf.greater(tf.shape(x['waveforms'])[0] / self.sr, self.minlen)
        )
        self.d = d.as_numpy_iterator()
        self.i += self.batch_files

    def __getitem__(self, idx, raise_exception=False):
        try:
            r = next(self.d)
        except Exception as e:
            if raise_exception:
                raise
            print('Exception __getitem__', e)
            self.get_dataset()
            r = next(self.d)
        r = {'speech': [r['waveforms']], 'sampling_rate': [16000],
             'target_text': ''.join([CTC_VOCAB[t] for t in r['targets']])}
        return r

    def __len__(self):
        return self.max_batch


with open('3mixed-train-test-v2.json') as fopen:
    dataset = json.load(fopen)

test_set = [
    'gs://mesolitica-tpu-general/mandarin/0-35.tfrecord',
    'gs://mesolitica-tpu-general/malay/2-25.tfrecord',
    'gs://mesolitica-tpu-general/singlish/2-34.tfrecord'
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[2]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)
    train_dataset = MalayaDataset(dataset['train'], directory='tfrecord-300m')
    eval_dataset = MalayaDataset(
        test_set,
        directory='tfrecord-300m-test',
        max_batch=100,
        overwrite_directory=False
    )

    vocab_dict = {v: k for k, v in enumerate(CTC_VOCAB)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("ctc-vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        "ctc-vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()

    def prepare_dataset(batch):
        inputs = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
    )
    eval_dataset = eval_dataset.map(
        prepare_dataset,
    )

    wer_metric = datasets.load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        print(list(zip(pred_str, label_str)))

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )

    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        print('checkpoint', checkpoint)

        processor.save_pretrained(training_args.output_dir)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = 100
        metrics["eval_samples"] = 100

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


if __name__ == "__main__":
    main()
