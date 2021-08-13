# Fastspeech2

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

**Fastspeech2 required alignment from Tacotron2**.

## how-to

1. Generate speech alignment from Tacotron2, 

Male speaker, [calculate-alignment-tacotron2-male-train.ipynb](calculate-alignment-tacotron2-male-train.ipynb) and [calculate-alignment-tacotron2-male-test.ipynb](calculate-alignment-tacotron2-male-test.ipynb]).

Female speaker, [calculate-alignment-tacotron2-female-train.ipynb](calculate-alignment-tacotron2-female-train.ipynb) and [calculate-alignment-tacotron2-female-test.ipynb](calculate-alignment-tacotron2-female-test.ipynb]).

Husein speaker, [calculate-alignment-tacotron2-husein.ipynb](calculate-alignment-tacotron2-husein.ipynb).

1. Run training script,

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