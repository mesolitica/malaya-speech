# SpeechSplit Conversion

Pretrained Malaya Speech SpeechSplit Conversion.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

## how-to

1. Download and prepare dataset, [prepare-dataset.ipynb](prepare-dataset.ipynb)

2. Train any models,

**FastSpeechSplit Pyworld**,

```bash
python3 fastspeechsplit-pyworld-crossentropy.py
```

3. Export the model for production, example for FastVC 32, [export/fastvc-32.ipynb](export/fastvc-32.ipynb)

## Download

1. FastSpeechSplit Pyworld, last update 30th May 2021, [fastspeechsplit-pyworld.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeechsplit-pyworld.tar.gz)

2. FastSpeechSplit V2 Pyworld, last update 30th May 2021, [fastspeechsplit-v2-pyworld.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeechsplit-v2-pyworld.tar.gz)

3. FastSpeechSplit PySPTK, last update 1st June 2021, [fastspeechsplit-pysptk.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeechsplit-pysptk.tar.gz)

4. FastSpeechSplit V2 PySPTK, last update 1st June 2021, [fastspeechsplit-v2-pysptk.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeechsplit-v2-pysptk.tar.gz)