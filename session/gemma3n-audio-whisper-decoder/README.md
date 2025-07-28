# gemma3n-audio-whisper-decoder

## how to replicate

### 1. Finetune Gemma3 Audio Encoder + Whisper Decoder

1. Prepare dataset, [prepare-mosaic.ipynb](prepare-mosaic.ipynb)

2. Finetune it,

```bash
bash alignment.sh
```

This finetune will freeze Gemma3 Audio Encoder.

### 2. Add VQ after projection

1. Generate codebook weight using K-means, [kmeans.ipynb](kmeans.ipynb)

2. Finetune it,

```bash
bash alignment-vq.sh
```

This finetune will freeze Gemma3 Audio Encoder but full parameter finetuning Whisper Decoder.