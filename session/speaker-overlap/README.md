# Speaker Overlap Detection

Finetuned available pretrained Malaya-Speech speaker vector models to do Speaker Overlap Detection.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

## how-to

1. Generate dataset [prepare/speaker-overlap-detection.ipynb](prepare/speaker-overlap-detection.ipynb).

2. Run any finetuning scripts,

**VGGVOX v2**,

```bash
python3 finetune-vggvox-v2.py
```

3. Export the model for production, example for vggvox-v2, [export/vggvox-v2.ipynb](export/vggvox-v2.ipynb)

## Download

1. VGGVox V2, last update 30th October 2020, [vggvox-v2-speaker-overlap-300k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/vggvox-v2-speaker-overlap-300k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/8pLkE1qWRPq49g3LrDfJsw/

2. SpeakerNet, last update 30th October 2020, [speakernet-speaker-overlap-300k.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/finetuned/speakernet-speaker-overlap-300k.tar.gz)

Tensorboard, https://tensorboard.dev/experiment/zPHBbBqHQbe1pNWYQTLh3Q/
