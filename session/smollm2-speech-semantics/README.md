# Finetune SmolLM2 for speech semantic tokens

## how to

1. Clone the dataset,

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/smollm2-speech-semantic-multipack-2048', repo_type='dataset', local_dir = './smollm2-speech-semantic-multipack-2048')
"
```

2. Finetune,

```bash
smollm2-135m-speech.sh
smollm2-360m-speech.sh
```