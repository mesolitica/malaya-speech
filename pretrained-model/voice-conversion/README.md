# Voice Conversion

Pretrained Malaya Speech Voice Conversion.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

## how-to

1. Download and prepare dataset, 

We use universal Vocoder data preparation, [../prepare-vocoder/prepare-universal.ipynb](../prepare-vocoder/prepare-universal.ipynb).

2. Train any models,

**FastVC 32**,

```bash
python3 fastvc-32.py
```

**FastVC 64**,

```bash
python3 fastvc-64.py
```

3. Export the model for production, example for FastVC 32, [export/fastvc-32.ipynb](export/fastvc-32.ipynb)

## Download

1. FastVC 32, last update 31st January 2021, [fastvc-32-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastvc-32-output.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/noreU9R0QRmbkWbbmiokvA/

2. FastVC 64, last update 31st January 2021, [fastvc-64-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastvc-64-output.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/wrrGljsQSOWvPDPk4HvK1A/

3. FastVC 32, last update 10st Feb 2021, [fastvc-32-output-v2.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastvc-32-output-v2.tar.gz)
