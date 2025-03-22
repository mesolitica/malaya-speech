# Orpheus

## dataset

Dataset at https://huggingface.co/datasets/mesolitica/TTS

### Prepare dataset

1. Convert speech to tokens,

```bash
python3 convert_snac.py \
--dataset 'mesolitica/TTS-Combined' --split 'train' --replication 3
```

Verify the speech tokens, [verify-speech-tokens.ipynb](verify-speech-tokens.ipynb).

2. Multipacking, [multipacking.ipynb](multipacking.ipynb).

Multipacking dataset pushed at [huseinzol05/orpheus-3k-multipacking](https://huggingface.co/datasets/huseinzol05/orpheus-3k-multipacking)

## how to finetune

### 3B

```bash
bash 3b.sh
```
