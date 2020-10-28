# Gender Detection

Finetuned available pretrained Malaya-Speech speaker vector models to do Gender Detection.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## how-to

1. Generate dataset [prepare/gender-detection.ipynb](prepare/gender-detection.ipynb).

2. Run any finetuning scripts,

**VGGVOX v2**,

```bash
python3 finetune-vggvox-v2.py
```

3. Export the model for production, example for vggvox-v2, [export/vggvox-v2.ipynb](export/vggvox.ipynb)

## Download

1. VGGVox V2, last update 21th October 2020, [vggvox-v2-gender-detection-175k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/vggvox-v2-gender-detection-175k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/ogQyGC3DTPeAHjkDOlxZnA/

2. Deep Speaker, last update 21th October 2020, [deep-speaker-gender-detection-100k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/deep-speaker-gender-detection-100k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/N60MVMzYSpOfotvfVYumXg/