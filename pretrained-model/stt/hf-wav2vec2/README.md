# HuggingFace Wav2Vec2

https://huggingface.co/docs/transformers/model_doc/wav2vec2, trained on https://github.com/huseinzol05/malaya-speech/tree/master/data/mixed-stt

**This directory is very lack of comments, able to understand HuggingFace, Torch Dataset, Tensorflow and Tensorflow Dataset are really helpful**.

## how-to pretrain

1. Pretrain `BASE` size,

```bash
python3 pretrain_v2.py config-pretrained-base.json
```

2. Pretrain `SMALL` size,

```bash
python3 pretrain_small.py config-pretrained-small.json
```

3. Pretrain `MINI` size,

```bash
python3 pretrain_mini.py config-pretrained-mini.json
```

## how-to finetune

### 300M model

1. Run finetune,

```bash
python3 finetune.py config-300m-v2.json
```

### 1B model

1. Run finetune,

```bash
python3 finetune.py config-1b.json
```

### 300M only [0, 1, 2, 3, 20, 21, 22, 23] layers

1. Save the model, [300m-8layers.ipynb](300m-8layers.ipynb).

2. Run finetune,

```bash
python3 train-300m-8layers.py config-300m-8layers.json
```

## how-to distill

1. Run teacher-student distillation,

```bash
python3 distill.py
```

## Run Tensorboard

```bash
CUDA_VISIBLE_DEVICES='' python3 -m tensorboard.main --logdir=runs --host=0.0.0.0
```

## download

1. Finetuned Wav2Vec2 XLS-R 300M on mixed language, https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-mixed
2. Pretrained Wav2Vec2 BASE on mixed language, https://huggingface.co/mesolitica/pretrained-wav2vec2-base-mixed
3. Pretrained Wav2Vec2 MINI on mixed language, https://huggingface.co/mesolitica/pretrained-wav2vec2-mini-mixed