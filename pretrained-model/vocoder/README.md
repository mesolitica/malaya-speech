# Vocoder

Pretrained Malaya Speech Vocoder models, Mel to Waveform.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

## how-to

1. Download and prepare dataset, 

For Male speaker, [../prepare-tts/preprocess-male.ipynb](../prepare-tts/preprocess-male.ipynb).

For Female speaker, [../prepare-tts/preprocess-female.ipynb](../prepare-tts/preprocess-female.ipynb).

For Husein speaker, [../prepare-tts/preprocess-husein.ipynb](../prepare-tts/preprocess-husein.ipynb).

2. Follow README inside model directory, example, [melgan](melgan).

## how-to-universal

Universal Vocoder able to synthesize multispeakers and zero-shot.

1. Download and prepare dataset, [../prepare-vocoder](../prepare-vocoder).

2. Follow README inside model directory, example, [universal-melgan](universal-melgan).




