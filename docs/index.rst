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

   load-cache
   gpu-environment
   Api
   Dataset
   Contributing

.. toctree::
   :maxdepth: 2
   :caption: Pipeline Module
   
   load-pipeline
   load-application-pipeline

.. toctree::
   :maxdepth: 2
   :caption: ASR Module

   load-stt-transducer-model
   transcribe-long-audio
   realtime-asr

.. toctree::
   :maxdepth: 2
   :caption: Vocoder Module

   load-vocoder
   load-universal-melgan

.. toctree::
   :maxdepth: 2
   :caption: Conversion Module

   load-voice-conversion
   
.. toctree::
   :maxdepth: 2
   :caption: TTS Module

   tts-tacotron2-model
   tts-fastspeech2-model
   more-tts-fastspeech2
   tts-long-text

.. toctree::
   :maxdepth: 2
   :caption: Classification Module

   load-age-detection
   load-emotion
   load-gender
   load-language-detection   
   load-speaker-overlap
   realtime-classification
   classification-stacking

.. toctree::
   :maxdepth: 2
   :caption: Enhancement Module
   
   load-noise-reduction
   load-speech-enhancement
   load-super-resolution

.. toctree::
   :maxdepth: 2
   :caption: Voice Activity Module

   load-vad
   realtime-vad
   split-utterances
   remove-silent-vad

.. toctree::
   :maxdepth: 2
   :caption: Speaker Diarization Module

   load-speaker-change
   load-diarization

.. toctree::
   :maxdepth: 2
   :caption: Speaker Vector Module

   load-speaker-vector
   pca-speaker

.. toctree::
   :maxdepth: 2
   :caption: Extra Module

   load-load-rttm

.. toctree::
   :maxdepth: 2
   :caption: Train ASR

   prepare-sebut-perkataan-tfrecord
   train-asr-cnn-rnn
   train-asr-mini-jasper
   train-asr-quartznet
   train-asr-small-conformer

.. toctree::
   :maxdepth: 2
   :caption: Misc

   Donation
