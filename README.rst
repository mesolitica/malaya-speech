.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="40%" src="https://i.imgur.com/ImYNHnm.png">
        </a>
    </p>
    <p align="center">
        <a href="https://pypi.python.org/pypi/malaya-speech"><img alt="Pypi version" src="https://badge.fury.io/py/malaya-speech.svg"></a>
        <a href="https://pypi.python.org/pypi/malaya-speech"><img alt="Python3 version" src="https://img.shields.io/pypi/pyversions/malaya-speech.svg"></a>
        <a href="https://github.com/huseinzol05/Malaya-Speech/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/huseinzol05/malaya-speech.svg?color=blue"></a>
        <a href="https://pepy.tech/project/malaya-speech"><img alt="total stats" src="https://static.pepy.tech/badge/malaya-speech"></a>
        <a href="https://pepy.tech/project/malaya-speech"><img alt="download stats / month" src="https://static.pepy.tech/badge/malaya-speech/month"></a>
        <a href="https://discord.gg/aNzbnRqt3A"><img alt="discord" src="https://img.shields.io/badge/discord%20server-malaya-rgb(118,138,212).svg"></a>
    </p>

=========

**Malaya-Speech** is a Speech-Toolkit library for Malaysian language, powered by PyTorch.

Documentation
--------------

Stable released documentation is available at https://malaya-speech.readthedocs.io/en/stable/

Installing from the PyPI
----------------------------------

::

    $ pip install malaya-speech

It will automatically install all dependencies except for PyTorch. So you can choose your own PyTorch CPU / GPU version.

Only **Python >= 3.6.0**, and **PyTorch >= 1.13** are supported.

If you are a Windows user, make sure read https://malaya.readthedocs.io/en/latest/running-on-windows.html

Development Release
---------------------------------

Install from `master` branch,
::

    $ pip install git+https://github.com/mesolitica/malaya-speech.git


We recommend to use **virtualenv** for development. 

While development released documentation is available at https://malaya-speech.readthedocs.io/en/latest/

Pretrained Models
------------------

Malaya-Speech also released Malaysian speech pretrained models, simply check at https://huggingface.co/mesolitica

References
-----------

If you use our software for research, please cite:

::

  @misc{Malaya-Speech, Speech-Toolkit library for Malaysian language, powered by PyTorch,
    author = {Husein, Zolkepli},
    title = {Malaya-Speech},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/mesolitica/malaya-speech}}
  }

Acknowledgement
----------------

Thanks to `KeyReply <https://www.keyreply.com/>`_ for private V100s cloud and `Mesolitica <https://mesolitica.com/>`_ for private RTXs cloud to train Malaya-Speech models,

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://image4.owler.com/logo/keyreply_owler_20191024_163259_original.png">
    </a>

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://i1.wp.com/mesolitica.com/wp-content/uploads/2019/06/Mesolitica_Logo_Only.png?fit=857%2C532&ssl=1">
    </a>