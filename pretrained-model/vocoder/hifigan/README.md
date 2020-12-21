# HifiGAN

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

## how-to

1. Pretrain generator first,

Male speaker,

```bash
python3 hifigan-male-generator.py
```

Female speaker,

```bash
python3 hifigan-female-generator.py
```

2. Train both generator and discriminator,

Male speaker,

```bash
python3 hifigan-male.py
```

Female speaker,

```bash
python3 hifigan-female.py
```
