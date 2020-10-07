# Speech Enchancement

Masked models and wavenet for Speech Enchancement.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## how-to

1. Run any training scripts,

**Resnet34-UNET**,

```bash
python3 resnet34-unet.py
```

**Inception-V3-UNET**,

```bash
python3 inception-v3-unet.py
```

2. Export the model for production, example for vggvox-v2, [export-vggvox-v2.ipynb](export-vggvox-v2.ipynb)

## Download

1. Resnet34-UNET, last update 7th October 2020, [output-resnet34-unet-50k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/output-resnet34-unet-50k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/Fd45NqegSRKHVi5Lj69RNg/#scalars