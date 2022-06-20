# LightSpeech

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

**LightSpeech required alignment from Tacotron2**.

## how-to

1. Generate speech alignment from Tacotron2, notebooks in [../fastspeech2/calculate-alignment](../fastspeech2/calculate-alignment). 

2. Run training script,

Osman speaker,

```bash
python3 lightspeech-osman.py
```

## download

1. Yasmin speaker, last update 13th June 2022, [yasmin-yasmin-output.tar](https://huggingface.co/huseinzol05/pretrained-lightspeech/blob/main/lightspeech-yasmin-output.tar).

  - Case sensitive, understand `.,?!` punctuations.

2. Osman speaker, last update 13th June 2022, [lightspeech-osman-output.tar](https://huggingface.co/huseinzol05/pretrained-lightspeech/blob/main/lightspeech-osman-output.tar).

  - Case sensitive, understand `.,?!` punctuations.