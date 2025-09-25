# Finetune Sesame TTS

## Prepare dataset

1. Trim silents, [trim-silence.ipynb](trim-silence.ipynb).

2. Convert to Moshi speech tokens,

```bash
python3 convert_moshi.py
```

3. Generate permutation voice conversion based on the same pseudospeaker label, [prepare-dataset.ipynb](prepare-dataset.ipynb).

4. Sort and multipack Decoder, [merge.ipynb](merge.ipynb).

Permutation, Moshi tokens and sorted multipacking pushed to [mesolitica/Malaysian-Emilia-Audio-Tokens](https://huggingface.co/datasets/mesolitica/Malaysian-Emilia-Audio-Tokens).

## Prepare podcast dataset

1. We post calculate speaker similarity using NEMO titanet-large for permutation voice conversion, [podcast/prepare-stage2.ipynb](podcast/prepare-stage2.ipynb), or run the script,

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 permutate.py --replication=20
```

2. After that we did the same silent trimming, convert to audio tokens and multipacking,

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 trim_silence.py --replication=20
```

## Finetuning

1. Run script,

```bash
bash finetune.sh
```