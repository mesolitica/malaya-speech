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

1. Clone the dataset first,

```bash
huggingface-cli download huseinzol05/orpheus-3k-multipacking --repo-type dataset --local-dir ./packing-3k 
```

2. Download the model without optimizer states,

```bash
HF_HOME="/workspace/cache" \
huggingface-cli download canopylabs/orpheus-3b-0.1-ft --exclude '*.bin*'
```

### 3B

```bash
bash 3b.sh
```
