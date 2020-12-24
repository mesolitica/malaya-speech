**Malaya-Speech** is a Speech-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow.

Documentation
--------------

Proper documentation is available at https://malaya-speech.readthedocs.io/

Installing from the PyPI
----------------------------------

CPU version
::

    $ pip install malaya-speech

GPU version
::

    $ pip install malaya-speech-gpu

Only **Python 3.6.x and above** and **Tensorflow 1.10 and above but not 2.0** are supported.

Features
--------
-  **Age Detection**

   Detect age in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Diarization**

   Diarizing speakers using Pretrained Speaker Vector Malaya-Speech models.
-  **Emotion Detection**

   Detect emotions in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Gender Detection**

   Detect genders in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Language Detection**

   Detect hyperlocal languages in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Noise Reduction**

   Reduce multilevel noises using Pretrained STFT UNET Malaya-Speech models.
-  **Speaker Change**

   Detect changing speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker overlap**

   Detect overlap speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Vector**

   Calculate similarity between speakers using Pretrained Malaya-Speech models.
-  **Speech Enhancement**

   Enhance voice activities using Pretrained STFT UNET Malaya-Speech models.
-  **Speech-to-Text**

   End-to-End Speech to Text using Pretrained CTC and RNN Transducer Malaya-Speech models.
-  **Text-to-Speech**

   End-to-End Text to Speech using Pretrained Tacotron2 and FastSpeech2 Malaya-Speech models.
-  **Vocoder**

   Convert Mel to Waveform using Pretrained MelGAN and Multiband MelGAN Vocoder Malaya-Speech models.
-  **Voice Activity Detection**

   Detect voice activities using Finetuned Speaker Vector Malaya-Speech models.

References
-----------

If you use our software for research, please cite:

::

  @misc{Malaya, Speech-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow,
    author = {Husein, Zolkepli},
    title = {Malaya-Speech},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/huseinzol05/malaya-speech}}
  }

Acknowledgement
----------------

Thanks to `Mesolitica <https://mesolitica.com/>`_ and `KeyReply <https://www.keyreply.com/>`_ for sponsoring GCP and private cloud to train Malaya models.