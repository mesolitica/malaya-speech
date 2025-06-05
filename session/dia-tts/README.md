# Finetune Dia TTS

## Prepare dataset

1. The dataset undergone silent trimming and generate permutation voice conversion from [mesolitica/Malaysian-Emilia-Audio-Tokens](https://huggingface.co/datasets/mesolitica/Malaysian-Emilia-Audio-Tokens), where this is done in [../sesame-tts](../sesame-tts/).

2. Convert to DAC speech tokens,

```bash
python3 convert_dac.py
```

3. Sort the permutation voice conversion and multipack for Encoder-Decoder, [merge-dia.ipynb](merge-dia.ipynb).

DAC tokens and sorted multipacking pushed to [mesolitica/Malaysian-Emilia-Audio-Tokens](https://huggingface.co/datasets/mesolitica/Malaysian-Emilia-Audio-Tokens).

## Finetuning

1. Run script,

```bash
bash finetune_dia_multipacking_v2.sh
```