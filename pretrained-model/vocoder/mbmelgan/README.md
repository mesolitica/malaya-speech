# Multiband MelGAN

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

## how-to

1. Pretrain generator first,

Male speaker,

```bash
python3 mbmelgan-male-generator.py
```

Female speaker,

```bash
python3 mbmelgan-male-generator.py
```

Husein speaker,

```bash
python3 mbmelgan-husein-generator.py
```

2. Train both generator and discriminator,

Male speaker,

```bash
python3 mbmelgan-male.py
```

Female speaker,

```bash
python3 mbmelgan-female.py
```

Husein speaker,

```bash
python3 mbmelgan-husein.py
```

## download

1. Female speaker, last update 23th December 2020, [mbmelgan-female-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/mbmelgan-female-output.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/BvNAV1f1T2mcV3u2Qpf18Q/

2. Male speaker, last update 23th December 2020, [mbmelgan-male-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/mbmelgan-male-output.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/qDnlEbpVRiKYvXEWpobTwA/

3. Husein speaker, last update 29th December 2020, [mbmelgan-husein-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/mbmelgan-husein-output.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/p0qAgIlpQAul837bF77sYA/