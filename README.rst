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

Thanks to,

1. `KeyReply <https://www.keyreply.com/>`_ for private V100s cloud.

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://image4.owler.com/logo/keyreply_owler_20191024_163259_original.png">
    </a>

2. `Nvidia <https://www.nvidia.com/en-us/>`_ for Azure credit.

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-horiz-500x200-2c50-d@2x.png">
    </a>


3. `Tensorflow Research Cloud <https://www.tensorflow.org/tfrc>`_ for free TPUs access.

.. raw:: html

    <a href="https://www.tensorflow.org/tfrc">
        <img alt="logo" width="20%" src="https://2.bp.blogspot.com/-xojf3dn8Ngc/WRubNXxUZJI/AAAAAAAAB1A/0W7o1hR_n20QcWyXHXDI1OTo7vXBR8f7QCLcB/s400/image2.png">
    </a>