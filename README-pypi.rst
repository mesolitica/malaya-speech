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

-  **Age Detection**, detect age in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Diarization**, diarizing speakers using Pretrained Speaker Vector Malaya-Speech models.
-  **Emotion Detection**, detect emotions in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Gender Detection**, detect genders in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Language Detection**, detect hyperlocal languages in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Noise Reduction**, reduce multilevel noises using Pretrained STFT UNET Malaya-Speech models.
-  **Speaker Change**, detect changing speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker overlap**, detect overlap speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Vector**, calculate similarity between speakers using Pretrained Malaya-Speech models.
-  **Speech Enhancement**, enhance voice activities using Pretrained STFT UNET Malaya-Speech models.
-  **Speech-to-Text**, End-to-End Speech to Text using Pretrained CTC and RNN Transducer Malaya-Speech models.
-  **Text-to-Speech**, using Pretrained Tacotron2 and FastSpeech2 Malaya-Speech models.
-  **Vocoder**, convert Mel to Waveform using Pretrained MelGAN and Multiband MelGAN Vocoder Malaya-Speech models.
-  **Voice Activity Detection**, detect voice activities using Finetuned Speaker Vector Malaya-Speech models.

Pretrained Models
------------------

Malaya-Speech also released pretrained models, simply check at `malaya-speech/pretrained-model <https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model>`_

- **Wave UNET**,  Multi-Scale Neural Network for End-to-End Audio Source Separation, https://arxiv.org/abs/1806.03185
- **Wave ResNet UNET**, added ResNet style into Wave UNET, no paper produced.
- **Deep Speaker**, An End-to-End Neural Speaker Embedding System, https://arxiv.org/pdf/1705.02304.pdf
- **SpeakerNet**, 1D Depth-wise Separable Convolutional Network for Text-Independent Speaker Recognition and Verification, https://arxiv.org/abs/2010.12653
- **VGGVox**, a large-scale speaker identification dataset, https://arxiv.org/pdf/1706.08612.pdf
- **GhostVLAD**, Utterance-level Aggregation For Speaker Recognition In The Wild, https://arxiv.org/abs/1902.10107
- **Conformer**, Convolution-augmented Transformer for Speech Recognition, https://arxiv.org/abs/2005.08100
- **ALConformer**, A lite Conformer, no paper produced.
-- **Jasper**, An End-to-End Convolutional Neural Acoustic Model, https://arxiv.org/abs/1904.03288
-- **Tacotron2**, Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions, https://arxiv.org/abs/1712.05884
-- **FastSpeech2**, Fast and High-Quality End-to-End Text to Speech, https://arxiv.org/abs/2006.04558
-- **MelGAN**, Generative Adversarial Networks for Conditional Waveform Synthesis, https://arxiv.org/abs/1910.06711
-- **Multi-band MelGAN**, Faster Waveform Generation for High-Quality Text-to-Speech, https://arxiv.org/abs/2005.05106

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