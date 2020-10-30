.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="50%" src="https://malaya-dataset.s3-ap-southeast-1.amazonaws.com/malaya-speech.png">
        </a>
    </p>
    <p align="center">
        <a href="https://pypi.python.org/pypi/malaya-speech"><img alt="Pypi version" src="https://badge.fury.io/py/malaya-speech.svg"></a>
        <a href="https://pypi.python.org/pypi/malaya-speech"><img alt="Python3 version" src="https://img.shields.io/pypi/pyversions/malaya-speech.svg"></a>
        <a href="https://github.com/huseinzol05/Malaya-Speech/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/huseinzol05/malaya-speech.svg?color=blue"></a>
    </p>

=========

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
-  **Speaker Diarization**

   Diarizing speakers using Pretrained Speaker Vector Malaya-Speech models.
-  **Age Detection**

   Detect age in speech using Finetuned Speaker Vector Malaya-Speech models.
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

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="50%" src="https://malaya-dataset.s3-ap-southeast-1.amazonaws.com/mesolitica-keyreply.png">
    </a>