# Fastpitch

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

**Fastpitch2 required alignment from Tacotron2**.

## how-to

1. Generate speech alignment from Tacotron2, 

Male speaker, [calculate-alignment-tacotron2-male-train.ipynb](../fastspeech2/calculate-alignment-tacotron2-male-train.ipynb) and [calculate-alignment-tacotron2-male-test.ipynb](../fastspeech2/calculate-alignment-tacotron2-male-test.ipynb]).

Female speaker, [calculate-alignment-tacotron2-female-train.ipynb](../fastspeech2/calculate-alignment-tacotron2-female-train.ipynb) and [../fastspeech2/calculate-alignment-tacotron2-female-test.ipynb](calculate-alignment-tacotron2-female-test.ipynb]).

Husein speaker, [calculate-alignment-tacotron2-husein.ipynb](../fastspeech2/calculate-alignment-tacotron2-husein.ipynb).

1. Run training script,

Female speaker,

```bash
python3 fastpitch-female.py
```

Male speaker,

```bash
python3 fastpitch-male.py
```

Husein speaker,

```bash
python3 fastpitch-husein.py
```

## download

1. Male speaker, last update 28th December 2020, [fastpitch-male-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastpitch-male-output.tar.gz)

  - Lower case, understand `.,?!` punctuations.

1. Female speaker, last update 28th December 2020, [fastpitch-male-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastpitch-male-output.tar.gz)

  - Lower case, understand `.,?!` punctuations.

1. Husein speaker, last update 28th December 2020, [fastpitch-male-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastpitch-male-output.tar.gz)

  - Lower case, understand `.,?!` punctuations.

1. Haqkiem speaker, last update 28th December 2020, [fastpitch-male-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastpitch-male-output.tar.gz)

  - Lower case, understand `.,?!` punctuations.

1. Female Singlish speaker, last update 28th December 2020, [fastpitch-female-singlish-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastpitch-female-singlish-output.tar.gz)

  - Lower case, understand `.,?!` punctuations.