# HuggingFace Whisper

**This directory is very lack of comments, able to understand HuggingFace, Torch Dataset, Tensorflow and Tensorflow Dataset are really helpful**.

## how-to finetune

### Whisper Tiny

1. Run finetune,

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 lightning.py \
--model='openai/whisper-tiny' \
--batch=24 \
--precision=16
```

### Whisper Base

1. Run finetune,

```bash
CUDA_VISIBLE_DEVICES=1 \
python3 lightning.py \
--model='openai/whisper-base' \
--batch=16 \
--precision=16
```

### Whisper Small

1. Run finetune,

```bash
CUDA_VISIBLE_DEVICES=1 \
python3 lightning.py \
--model='openai/whisper-small' \
--batch=12 \
--precision=16 \
--gradient_checkpoint=1
```

## download

1. https://huggingface.co/mesolitica/finetune-whisper-base-ms-singlish-v2
2. https://huggingface.co/mesolitica/finetune-whisper-tiny-ms-singlish-v2
3. https://huggingface.co/mesolitica/finetune-whisper-tiny-ms-singlish
4. https://huggingface.co/mesolitica/finetune-whisper-small-ms-singlish-v2
5. https://huggingface.co/mesolitica/finetune-whisper-base-ms-singlish