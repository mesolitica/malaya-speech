# Age Detection

Finetuned available pretrained Malaya-Speech speaker vector models to do Age Detection.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## how-to

1. Generate dataset [prepare-age-detection.ipynb](prepare-age-detection.ipynb).

2. Run any finetuning scripts,

**VGGVOX v2**,

```bash
python3 finetune-vggvox-v2.py
```

3. Export the model for production, example for vggvox-v2, [export-vggvox-v2-age-detection.ipynb](export-vggvox-v2-age-detection.ipynb)

## Download

1. VGGVox V2, last update 21th October 2020, [vggvox-v2-age-detection-300k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/vggvox-v2-age-detection-300k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/G3yFzR36Sruxx13Om2iOng/

2. Deep Speaker, last update 21th October 2020, [deep-speaker-age-detection-300k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/deep-speaker-age-detection-300k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/s85V0NK1T4q5xFd9zFzI3Q/