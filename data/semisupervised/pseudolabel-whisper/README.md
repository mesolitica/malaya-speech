# distributed multi-GPUs pseudolabel using Whisper

This pseudolabel included fast hashing load audio files and continue last step decoded.

## how-to

1. Configure accelerate,

```bash
accelerate config
```

2. Run accelerate,

```bash
~/my-env/bin/accelerate launch run.py --indices_filename=global-indices.json --batch_size=4
```