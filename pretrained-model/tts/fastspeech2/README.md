# Fastspeech2

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

**Fastspeech2 required alignment from Tacotron2**.

## how-to

1. Generate speech alignment from Tacotron2, notebooks in [calculate-alignment](calculate-alignment). 

2. Run training script,

Female speaker,

```bash
python3 fastspeech2-female.py
```

Male speaker,

```bash
python3 fastspeech2-male.py
```

Husein speaker,

```bash
python3 fastspeech2-husein.py
```

## download

1. Male speaker, last update 28th December 2020, [fastspeech2-male-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-male-output.tar.gz)

  - Tensorboard, https://tensorboard.dev/experiment/ZOY3C4u0SmOPI4ImPSWpCw/
  - Lower case, ignore punctuations.

2. Male speaker V2, last update 30th December 2020, [fastspeech2-male-v2-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-male-output.tar.gz)

  - Tensorboard, https://tensorboard.dev/experiment/6qZzTKNQT6OdCKKJ3V4SdA/
  - Lower case, ignore punctuations.

3. Husein speaker, last update 29th December 2020, [fastspeech2-husein-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-husein-output.tar.gz)

  - Tensorboard, https://tensorboard.dev/experiment/kvajI3r4TQeOFovUXqg8Fg/
  - Lower case, ignore punctuations.

4. Husein speaker V2, last update 29th December 2020, [fastspeech2-husein-v2-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-husein-v2-output.tar.gz)

  - Tensorboard, https://tensorboard.dev/experiment/foxnL7YHQrSL5oQZteO4Yw/
  - Lower case, ignore punctuations.

5. Female speaker, last update 29th December 2020, [fastspeech2-female-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-female-output.tar.gz)

  - Tensorboard, https://tensorboard.dev/experiment/nvVkoiamRhasVU4xvOh2Ag/
  - Lower case, ignore punctuations.

6. Female speaker V2, last update 29th December 2020, [fastspeech2-female-v2-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-female-v2-output.tar.gz)

  - Tensorboard, https://tensorboard.dev/experiment/58O7cjX6QdWM9qGooXMmjQ/
  - Lower case, ignore punctuations.

7. Haqkiem speaker, last update 7th January 2021, [fastspeech2-haqkiem-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-haqkiem-output.tar.gz)

  - Tensorboard, https://tensorboard.dev/experiment/skMZrZnLQpyOiG35bO0Mmw/
  - Lower case, understand `.,?!` punctuations.

8. Female Singlish speaker, last update 9th April 2021, [fastspeech2-female-singlish-output.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-female-singlish-output.tar.gz)

  - Lower case, understand `.,?!` punctuations.

9. Male speaker, last update 10th August 2021, [fastspeech2-male-output-2021-08-10.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-male-output-2021-08-10.tar.gz)

  - Lower case, understand `.,?!` punctuations.

10. Female speaker, last update 10th August 2021, [fastspeech2-female-output-2021-08-10.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-female-output-2021-08-10.tar.gz)

  - Lower case, understand `.,?!` punctuations.

11. Husein speaker, last update 10th August 2021, [fastspeech2-husein-output-2021-08-10.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-husein-output-2021-08-10.tar.gz)

  - Lower case, understand `.,?!` punctuations.

12. Female Singlish speaker, last update 22nd September 2021, [fastspeech2-female-singlish-output-v2.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/fastspeech2-female-singlish-output-v2.tar.gz)

  - Lower case, understand `.,?!` punctuations.

13. Yasmin speaker, last update 29th April 2022, [fastspeech2-small-yasmin-output.tar](https://huggingface.co/huseinzol05/pretrained-fastspeech2/resolve/main/fastspeech2-small-yasmin-output.tar).

  - Case sensitive, understand `.,?!` punctuations.
  - Small model.

14. Yasmin speaker, last update 29th April 2022, [fastspeech2-yasmin-output.tar](https://huggingface.co/huseinzol05/pretrained-fastspeech2/resolve/main/fastspeech2-yasmin-output.tar).

  - Case sensitive, understand `.,?!` punctuations.

15. Osman speaker, last update 29th April 2022, [fastspeech2-small-osman-output.tar](https://huggingface.co/huseinzol05/pretrained-fastspeech2/resolve/main/fastspeech2-small-osman-output.tar).

  - Case sensitive, understand `.,?!` punctuations.
  - Small model.

16. Osman speaker, last update 29th April 2022, [fastspeech2-osman-output.tar](https://huggingface.co/huseinzol05/pretrained-fastspeech2/resolve/main/fastspeech2-osman-output.tar).

  - Case sensitive, understand `.,?!` punctuations.