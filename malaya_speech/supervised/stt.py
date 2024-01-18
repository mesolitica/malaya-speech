from malaya_speech.utils import check_file
from malaya_boilerplate.huggingface import download_files
from malaya_speech.utils.read import load as load_wav
from malaya_speech.torch_model.huggingface import (
    CTC as HuggingFace_CTC,
    Aligner as HuggingFace_Aligner,
    Seq2Seq as HuggingFace_Seq2Seq,
    Seq2SeqAligner as HuggingFace_Seq2SeqAligner,
)
from malaya_speech.torch_model.torchaudio import Transducer, ForceAlignment
from transformers import AutoModelForCTC, AutoProcessor, AutoModelForSpeechSeq2Seq
from malaya_speech.path import TRANSDUCER_VOCABS, TRANSDUCER_MIXED_VOCABS
import torch
import os


def get_vocab(language):
    return TRANSDUCER_VOCABS.get(language, TRANSDUCER_VOCABS['malay'])


def get_vocab_mixed(language):
    return TRANSDUCER_MIXED_VOCABS.get(language)


dummy_sentences = ['tangan aku disentuh lembut', 'sebut perkataan angka']
default_reduction_factor = 4
time_reduction_factor = {
    'tiny-conformer': 4,
    'small-conformer': 4,
    'conformer': 4,
    'large-conformer': 4,
    'alconformer': 4,
    'xs-squeezeformer': 4,
    'sm-squeezeformer': 4,
    'm-squeezeformer': 4,
}


def ctc(model, stt=True, **kwargs):

    hf_model = AutoModelForCTC.from_pretrained(model, **kwargs)

    if stt:
        selected_model = HuggingFace_CTC
    else:
        selected_model = HuggingFace_Aligner

    return selected_model(
        hf_model=hf_model,
        model=model,
        name='speech-to-text'
    )


def seq2seq(model, stt=True, **kwargs):

    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(model, **kwargs)
    processor = AutoProcessor.from_pretrained(model, **kwargs)

    if stt:
        selected_model = HuggingFace_Seq2Seq
    else:

        selected_model = HuggingFace_Seq2SeqAligner

    return selected_model(
        hf_model=hf_model,
        processor=processor,
        model=model,
        name='speech-to-text',
        **kwargs,
    )


def torchaudio(model, stt=True, **kwargs):
    s3_file = {
        'model': 'model.pt',
        'sp_model': 'malay-stt.model',
        'stats_file': 'malay-stats.json',
    }
    path = download_files(model, s3_file, **kwargs)

    if stt:
        selected_model = Transducer
        name = 'speech-to-text'
    else:
        selected_model = ForceAlignment
        name = 'force-alignment'

    return selected_model(
        pth=path['model'],
        sp_model=path['sp_model'],
        stats_file=path['stats_file'],
        model=model,
        name=name,
    )
