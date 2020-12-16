# Vocoder

Pretrained MelGAN, MBMelGAN and HifiGAN mel to wav.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## how-to

1. Download and prepare Male dataset, [../prepare-tts/preprocess-male.ipynb](../prepare-tts/preprocess-male.ipynb).

2. Download and prepare Female dataset, [../prepare-tts/preprocess-female.ipynb](../prepare-tts/preprocess-female.ipynb).

### MelGAN

#### Male

1. Run [melgan-male.py](melgan-male.py).

#### Female

1. Run [melgan-female.py](melgan-female.py).

### MB-MelGAN

#### Male

1. Train generator first for 200k steps, [mbmelgan-male-generator.py](mbmelgan-male-generator.py).

2. Train generator and discriminator, [mbmelgan-male.py](mbmelgan-male.py).

#### Female

1. Train generator first for 200k steps, [mbmelgan-female-generator.py](mbmelgan-female-generator.py).

2. Train generator and discriminator, [mbmelgan-female.py](mbmelgan-female.py).

### HifiGAN

#### Male

1. Train generator first for 100k steps, [hifigan-male-generator.py](hifigan-male-generator.py).

2. Train generator and discriminator, [hifigan-male.py](hifigan-male.py).

#### Female

1. Train generator first for 100k steps, [hifigan-female-generator.py](hifigan-female-generator.py).

2. Train generator and discriminator, [hifigan-female.py](hifigan-female.py).

### Tips

1. Make sure always step generator first, after that discriminator.