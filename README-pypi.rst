**Malaya-Speech** is a Speech-Toolkit library for Malaysian language, powered by PyTorch.

Documentation
--------------

Stable released documentation is available at https://malaya-speech.readthedocs.io/

Installing from the PyPI
----------------------------------

::

    $ pip install malaya-speech

It will automatically install all dependencies except for Tensorflow and PyTorch. So you can choose your own Tensorflow CPU / GPU version and PyTorch CPU / GPU version.

Only **Python >= 3.6.0**, **Tensorflow >= 1.15.0**, and **PyTorch >= 1.10** are supported.

Development Release
---------------------------------

Install from `master` branch,
::

    $ pip install git+https://github.com/huseinzol05/malaya-speech.git


We recommend to use **virtualenv** for development. 

Documentation at https://malaya-speech.readthedocs.io/en/latest/

Pretrained Models
------------------

Malaya-Speech also released Malaysian speech pretrained models, simply check at https://huggingface.co/mesolitica

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

Thanks to `KeyReply <https://www.keyreply.com/>`_ for private V100s cloud and `Mesolitica <https://mesolitica.com/>`_ for private RTXs cloud to train Malaya models.

Also, thanks to `Tensorflow Research Cloud <https://www.tensorflow.org/tfrc>`_ for free TPUs access.