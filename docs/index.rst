.. malaya documentation master file, created by
   sphinx-quickstart on Sat Dec  8 23:44:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Malaya-Speech's documentation!
==========================================

.. include::
   README.rst

Contents:
=========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   mock-tensorflow
   Dataset
   Contributing

.. toctree::
   :maxdepth: 2
   :caption: GPU Environment

   gpu-environment-tensorflow
   gpu-environment-pytorch

.. toctree::
   :maxdepth: 2
   :caption: Pipeline Module
   
   load-pipeline
   load-application-pipeline

.. toctree::
   :maxdepth: 2
   :caption: Language Model Module
   
   kenlm
   gpt2-lm
   masked-lm

.. toctree::
   :maxdepth: 2
   :caption: ASR RNNT Module

   load-stt-transducer-model
   load-stt-transducer-model-lm
   load-stt-transducer-model-lm-gpt2
   load-stt-transducer-model-lm-mlm
   load-stt-transducer-model-singlish
   load-stt-transducer-model-2mixed
   load-stt-transducer-model-pt

.. toctree::
   :maxdepth: 2
   :caption: ASR CTC Module

   load-stt-ctc-model
   load-stt-ctc-model-ctc-decoders
   load-stt-ctc-model-pyctcdecode
   load-stt-ctc-model-pyctcdecode-gpt2
   load-stt-ctc-model-pyctcdecode-mlm
   stt-ctc-huggingface
   stt-ctc-huggingface-ctc-decoders
   stt-ctc-huggingface-pyctcdecode

.. toctree::
   :maxdepth: 2
   :caption: ASR Seq2Seq Module

   stt-seq2seq-whisper
   stt-seq2seq-huggingface
   
.. toctree::
   :maxdepth: 2
   :caption: Force Alignment Module

   force-alignment
   force-alignment-ctc
   force-alignment-huggingface
   put-comma-force-alignment

.. toctree::
   :maxdepth: 2
   :caption: Vocoder Module

   load-vocoder
   load-universal-melgan
   load-universal-hifigan

.. toctree::
   :maxdepth: 2
   :caption: Conversion Module

   load-voice-conversion
   speechsplit-conversion-pyworld
   speechsplit-conversion-pysptk
   
.. toctree::
   :maxdepth: 2
   :caption: TTS Module

   tts-tacotron2-model
   tts-fastspeech2-model
   more-tts-fastspeech2
   fastspeech2-long-text
   tts-e2e-fastspeech2
   tts-fastpitch-model
   tts-glowtts-model
   tts-glowtts-multispeaker-model
   tts-lightspeech-model
   tts-vits
   vits-long-text
   tts-singlish
   tts-gradio

.. toctree::
   :maxdepth: 2
   :caption: Classification Module

   load-age-detection
   load-emotion
   load-gender
   load-is-clean
   load-language-detection   
   load-speaker-overlap
   classification-stacking

.. toctree::
   :maxdepth: 2
   :caption: Enhancement Module
   
   load-noise-reduction
   load-speech-enhancement
   load-super-resolution-unet
   load-super-resolution-tfgan
   load-super-resolution-audio-diffusion

.. toctree::
   :maxdepth: 2
   :caption: Voice Activity Module

   load-vad
   split-utterances
   remove-silent-vad

.. toctree::
   :maxdepth: 2
   :caption: Speaker Vector Module

   load-speaker-vector
   load-speaker-vector-nemo
   load-speaker-vector-huggingface

.. toctree::
   :maxdepth: 2
   :caption: Speaker Diarization Module

   load-speaker-change
   load-diarization-speaker-similarity
   load-diarization-clustering
   load-diarization-clustering-agglomerative
   load-diarization-clustering-hmm
   load-diarization-speaker-change
   load-diarization-timestamp
   load-diarization-using-features
   combine-longer-speaker-diarization

.. toctree::
   :maxdepth: 2
   :caption: PyAudio streaming Module
   
   realtime-vad
   realtime-asr
   realtime-asr-without-vad
   realtime-alignment
   realtime-classification

.. toctree::
   :maxdepth: 2
   :caption: TorchAudio streaming Module
   
   long-audio-vad-torchaudio
   long-audio-asr-torchaudio
   rnnt-streaming-torchaudio
   long-audio-classification-torchaudio
   youtube-asr-diarization-torchaudio

.. toctree::
   :maxdepth: 2
   :caption: Multispeaker Module

   multispeaker-separation-wav

.. toctree::
   :maxdepth: 2
   :caption: Extra Module

   load-load-rttm
   pca-speaker

.. toctree::
   :maxdepth: 2
   :caption: Misc

   Api
   Donation
