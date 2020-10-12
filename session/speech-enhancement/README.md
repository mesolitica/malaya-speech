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

1. Resnet34-UNET, last update 7th October 2020, [resnet34-unet-100k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/resnet34-unet-100k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/OjXkwPdqReOB4zEpYRT3sQ/

2. Inception V3 UNET, last update 11th October 2020, [inception-v3-unet.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/inception-v3-unet.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/Cw8vnAM1Q5GTOZNwsNdmng/