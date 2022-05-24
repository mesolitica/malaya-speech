# HuggingFace Wav2Vec2

https://huggingface.co/docs/transformers/model_doc/wav2vec2, trained on https://github.com/huseinzol05/malaya-speech/tree/master/data/mixed-stt

**This directory is very lack of comments, able to understand HuggingFace, Torch Dataset, Tensorflow, Tensorflow Dataset is really helpful**.

## how-to

### 300M model

1. Run finetune,

```bash
python3 train-300m.py config-300m.json
```

### 1B model

1. Run finetune,

```bash
python3 train-1b.py config-1b.json
```

### 300M last 8 layers

1. Save the model, [300m-8layers.ipynb](300m-8layers.ipynb).

2. Run finetune,

```bash
python3 train-300m-8layers.py config-300m-8layers.json
```

## download

1. https://huggingface.co/malay-huggingface/wav2vec2-xls-r-300m-mixed
2. https://huggingface.co/malay-huggingface/wav2vec2-xls-r-1b-mixed