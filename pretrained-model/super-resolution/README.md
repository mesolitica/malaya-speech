# Super Resolution

Pretrained SRGAN to do audio super resolution.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

## how-to

1. Run any training scripts,

**SRGAN-256**,

```bash
python3 srgan-256.py
```

**SRGAN-128**,

```bash
python3 srgan-128.py
```

2. Export the model for production, example for UNET, [export/srgan-256.ipynb](export/srgan-256.ipynb)

## Download

1. SRGAN 256, last update 19th January 2021, [srgan-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/srgan-output.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/R5NrQXNxRZGTrO25eiE8sw/

2. SRGAN 128, last update 19th January 2021, [srgan-128-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/srgan-128-output.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/mDRHWoVkT7uMdfgbp2cpJQ/