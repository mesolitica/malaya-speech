# Speech Enchancement

Finetuned Malaya-Speech UNET models to do Speech Enhancement.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

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

3. UNET-24, last update 21st January 2021, [speech-enhancement-24.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/speech-enhancement-24.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/yBxCZgaCRkOxvEJIjHgsPw/

3. UNET-36, last update 21st January 2021, [speech-enhancement-36.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/speech-enhancement-36.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/RQ0ZKeldSKy664MdSl4xEA/