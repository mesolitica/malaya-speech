# HuggingFace Wav2Vec2

https://huggingface.co/docs/transformers/model_doc/wav2vec2, trained on https://github.com/huseinzol05/malaya-speech/tree/master/data/mixed-stt

**This directory is very lack of comments, able to understand HuggingFace, Torch Dataset, Tensorflow and Tensorflow Dataset are really helpful**.

## how-to finetune

### XLS-R-300M

1. Run finetune,

```bash
CUDA_VISIBLE_DEVICES=0 python3 lightning.py \
--model='facebook/wav2vec2-xls-r-300m' --batch=12 --precision=32 --learning_rate=5e-5 --gradient_checkpoint=1
```

### XLS-R-300M 12 layers

1. copy to smaller model, [prepare-300m-12layers.ipynb](prepare-300m-12layers.ipynb).

2. Run finetune,

```bash
CUDA_VISIBLE_DEVICES=1 python3 wav2vec2-lightning-ms.py \                                         
--model='300m-12-layers' --batch=12 --precision=32 --learning_rate=5e-5 --gradient_checkpoint=1
```

### XLS-R-300M 6 layers

1. copy to smaller model, [prepare-300m-6layers.ipynb](prepare-300m-6layers.ipynb).

2. Run finetune,

```bash
CUDA_VISIBLE_DEVICES=1 python3 wav2vec2-lightning-ms.py \                                         
--model='300m-6-layers' --batch=12 --precision=32 --learning_rate=5e-5
```

### XLS-R-300M 3 layers

1. copy to smaller model, [prepare-300m-3layers.ipynb](prepare-300m-3layers.ipynb).

2. Run finetune,

```
CUDA_VISIBLE_DEVICES=0 python3 wav2vec2-lightning-ms.py \
--model='300m-3-layers' --batch=12 --precision=32 --learning_rate=5e-5
```

## download

1. https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-12layers-ms
2. https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-mixed
3. https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-mixed-v2
4. https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-6layers-ms
5. https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-3layers-ms