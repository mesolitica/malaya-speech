# distributed multi-GPUs pseudolabel using Whisper

This pseudolabel included fast hashing load audio files and continue last step decoded.

## how-to

### Use accelerate

1. Configure accelerate,

```bash
accelerate config
```

2. Run accelerate,

```bash
~/my-env/bin/accelerate launch run.py --indices_filename=global-indices.json --batch_size=4
```

### Use torchrun

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 ~/my-env/bin/torchrun --nproc_per_node 2 \
-m run \
--indices_filename=global-indices.json --batch_size=4
```

NCCL is not required.