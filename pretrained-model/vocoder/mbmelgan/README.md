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

2. Train both generator and discriminator,

Male speaker,

```bash
python3 mbmelgan-male.py
```

Female speaker,

```bash
python3 mbmelgan-female.py
```
