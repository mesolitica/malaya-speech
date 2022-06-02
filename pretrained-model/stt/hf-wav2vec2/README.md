# HuggingFace Wav2Vec2

https://huggingface.co/docs/transformers/model_doc/wav2vec2, trained on https://github.com/huseinzol05/malaya-speech/tree/master/data/mixed-stt

**This directory is very lack of comments, able to understand HuggingFace, Torch Dataset, Tensorflow, Tensorflow Dataset is really helpful**.

## how-to

### 300M model

1. Run finetune,

```bash
python3 train-300m-v2.py config-300m-v2.json
```

### 1B model

1. Run finetune,

```bash
python3 train-1b.py config-1b.json
```

### 300M only [0, 1, 2, 3, 20, 21, 22, 23] layers

1. Save the model, [300m-8layers.ipynb](300m-8layers.ipynb).

2. Run finetune,

```bash
python3 train-300m-8layers.py config-300m-8layers.json
```

### Run Tensorboard

```bash
CUDA_VISIBLE_DEVICES='' python3 -m tensorboard.main --logdir=runs --host=0.0.0.0
```

## download

1. https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-mixed