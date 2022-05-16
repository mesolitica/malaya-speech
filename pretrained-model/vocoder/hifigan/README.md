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

## download

1. Male, last update 29th October 2021, [hifigan-male-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/hifigan-male-output.tar.gz)

2. Female, last update 29th October 2021, [hifigan-female-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/hifigan-female-output.tar.gz)

3. Universal, last update 29th October 2021, [universal-hifigan-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/universal-hifigan-output.tar.gz)

4. Universal 512, last update 29th October 2021, [universal-hifigan-512-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/universal-hifigan-512-output.tar.gz)

5. Universal 1024, last update 16th May 2022, [universal-hifigan-1024-output.tar](https://huggingface.co/huseinzol05/pretrained-vocoder/resolve/main/universal-hifigan-1024-output.tar)