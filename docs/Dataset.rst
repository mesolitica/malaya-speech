.. role:: raw-html-m2r(raw)
   :format: html


Speech Dataset
==============

How we gather dataset?
----------------------


#. For semisupervised transcript, we use Google Speech to Text, after that verified / corrected by human.
#. We recorded using our own microphones.

License
-------

Malay-Speech dataset is available to download for research purposes under a Creative Commons Attribution 4.0 International License.

:raw-html-m2r:`<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>`\ :raw-html-m2r:`<br />`\ This work is licensed under a :raw-html-m2r:`<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>`.

Dataset
-------

`Ambient <https://github.com/huseinzol05/malaya-speech/tree/master/data/ambient>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple ambients gathered from Youtube.

`Audiobook <https://github.com/huseinzol05/malaya-speech/tree/master/data/audiobook>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gather Youtube urls for indonesian, english and low quality english audiobooks only.

`IIUM <https://github.com/huseinzol05/malaya-speech/tree/master/data/iium>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read random sentences from IIUM Confession.


* voice by `Husein Zolkepli <https://www.linkedin.com/in/husein-zolkepli/>`_ and `Shafiqah Idayu <https://www.facebook.com/shafiqah.ayu>`_.
* Heavily speaking in Selangor dialect.
* Recorded using low-end tech microphone.
* 44100 sample rate, split by end of sentences.
* approximate 2.4 hours.
* Still on going recording.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Speech Dataset from IIUM Confession texts,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/iium}}
   }

`IIUM-Clear <https://github.com/huseinzol05/malaya-speech/tree/master/data/iium-clear>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read random sentences from IIUM Confession, cleaner version.


* voice by `Husein Zolkepli <https://www.linkedin.com/in/husein-zolkepli/>`_.
* Heavily speaking in Selangor dialect.
* Recorded using mid-end tech microphone.
* 44100 sample rate, random 7 - 11 words window.
* approximate 0.1 hours.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Speech Dataset from IIUM Confession texts,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/iium}}
   }

`IMDA <https://github.com/huseinzol05/malaya-speech/tree/master/data/imda>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mirror link for IMDA dataset, https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus, only downloaded PART 3 and SST dataset.


* 16000 sample rate.
* supervised approximate 2024 hours.

`language <https://github.com/huseinzol05/malaya-speech/tree/master/data/language>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gather youtube urls for hyperlocal language detection from speech {malay, indonesian, manglish, english, mandarin}.

Check hyperlocal language detection models at https://malaya-speech.readthedocs.io/en/latest/load-language-detection.html

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Hyperlocal languages for speech dataset,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/language}}
   }

`news <https://github.com/huseinzol05/malaya-speech/tree/master/data/news>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read random sentences from bahasa news.


* voice by `Husein Zolkepli <https://www.linkedin.com/in/husein-zolkepli/>`_.
* Heavily speaking in Selangor dialect.
* Recorded using mid-end tech microphone, suitable for text to speech.
* 44100 sample rate, random 7 - 11 words window.
* approximate 3.01 hours.
* Still on going recording.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Speech Dataset from local news texts,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/news}}
   }

`noise <https://github.com/huseinzol05/malaya-speech/tree/master/data/noise>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple noises gathered from Youtube.

`Sebut perkataan <https://github.com/huseinzol05/malaya-speech/tree/master/data/sebut-perkataan>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read random words from malay dictionary started with 'tolong sebut :raw-html-m2r:`<word>`\ '.


* ``sebut-perkataan-man`` voice by `Husein Zolkepli <https://www.linkedin.com/in/husein-zolkepli/>`_
* ``tolong-sebut`` voice by `Khalil Nooh <https://www.linkedin.com/in/khalilnooh/>`_
* ``sebut-perkataan-woman`` voice by `Mas Aisyah Ahmad <https://www.linkedin.com/in/mas-aisyah-ahmad-b46508a9/>`_
* Recorded using low-end tech microphones.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Short Speech Dataset,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/sebut-perkataan}}
   }

`Semisupervised audiobook <https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-audiobook>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Semisupervised malay audiobooks from Nusantara Audiobook using Google Speech to Text.


* 44100 sample rate, super clean.
* semisupervised approximate 45.29 hours.
* windowed using Malaya-Speech VAD, each atleast 5 negative voice activities.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Semisupervised Speech Recognition from Audiobook,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-audiobook}}
   }

`Semisupervised malay <https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-malay>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Semisupervised malay youtube videos using Google Speech to Text, after that corrected by human.


* 16000 sample rate.
* semisupervised approximate 1804 hours.
* random length between 2 - 20 seconds, windowed using google VAD.
* supervised 768 samples, approximate 1.3 hours.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Semisupervised Speech Recognition from Malay Youtube Videos,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-malay}}
   }

`Semisupervised manglish <https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-manglish>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Semisupervised manglish youtube videos using Google Speech to Text.


* 16000 sample rate.
* semisupervised approximate 107 hours.
* random length between 2 - 20 seconds, windowed using google VAD.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Semisupervised Speech Recognition from Manglish Youtube Videos,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/semisupervised-manglish}}
   }

`wattpad <https://github.com/huseinzol05/malaya-speech/tree/master/data/wattpad>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read random sentences from bahasa wattpad.


* voice by `Husein Zolkepli <https://www.linkedin.com/in/husein-zolkepli/>`_.
* Heavily speaking in Selangor dialect.
* Recorded using mid-end tech microphone, suitable for text to speech.
* 44100 sample rate, random 7 - 11 words window.
* approximate 0.15 hours.
* Still on going recording.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Speech Dataset from Wattpad texts,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/wattpad}}
   }

`Wikipedia <https://github.com/huseinzol05/malaya-speech/tree/master/data/wikipedia>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read random sentences from Bahasa Wikipedia.


* voice by `Husein Zolkepli <https://www.linkedin.com/in/husein-zolkepli/>`_.
* Heavily speaking in Selangor dialect.
* Recorded using low-end tech microphone.
* 44100 sample rate, 4 words window.
* approximate 3.4 hours.
* Still on going recording.

.. code-block:: bibtex

   @misc{Malay-Dataset, We gather Bahasa Malaysia corpus!, Speech Dataset from Wikipedia texts,
     author = {Husein, Zolkepli},
     title = {Malay-Dataset},
     year = {2018},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/huseinzol05/malaya-speech/tree/master/data/wikipedia}}
   }

`youtube <https://github.com/huseinzol05/malaya-speech/tree/master/data/youtube>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gathered Jeorogan, malay, malaysian, the thirsty sisters, richroll podcasts.

Contribution
------------

Contact us at husein.zol05@gmail.com or husein@mesolitica.com if want to contribute to speech bahasa dataset.
