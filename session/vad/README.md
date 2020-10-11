# Voice Activity Detection

Finetuned available pretrained Malaya-Speech speaker vector models to do Voice Activity Detection.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## how-to

1. Generate dataset [prepare-vad-dataset.ipynb](prepare-vad-dataset.ipynb).

2. Run any finetuning scripts,

**VGGVOX v1**,

```bash
python3 finetune-vggvox-v1.py
```

**VGGVOX v2**,

```bash
python3 finetune-vggvox-v2.py
```

3. Export the model for production, example for vggvox-v2, [export-vggvox-v2.ipynb](export-vggvox-v2.ipynb)

## Download

1. VGGVox V2, last update 24th September 2020, [output-vggvox-v2-vad-300k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/output-vggvox-v2-vad-300k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/Fd45NqegSRKHVi5Lj69RNg/

2. VGGVox V1, last update 7th October 2020, [vggvox-v1-vad-170k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/vggvox-v1-vad-170k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/Ej1t6x9sQ8CPdiw1fv4lew/

3. Inception V4, last update 10th October 2020, [inception-v4-vad-300k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/inception-v4-vad-300k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/UJvCrLsyQjSlS3OrdXBe4Q/