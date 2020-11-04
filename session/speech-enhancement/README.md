# Speech Enchancement

Finetuned Malaya-Speech UNET models to do Speech Enhancement.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## how-to

1. Run any training scripts,

**Resnet-UNET**,

```bash
python3 resnet-unet.py
```

**UNET**,

```bash
python3 unet.py
```

2. Export the model for production, example for UNET, [export/unet.ipynb](export/unet.ipynb)

## Download

1. UNET, last update 3rd November 2020, [speech-enhancement-unet-output-500k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/speech-enhancement-unet-output-500k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/nq6XKjhxQfKDsKrpyys6iA/

2. RESNET-UNET, last update 4rd November 2020, [speech-enhancement-resnet-unet-output-500k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/speech-enhancement-resnet-unet-output-500k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/v446IlUvRMq1PhQ9JqfABg/