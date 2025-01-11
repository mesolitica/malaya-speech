# Whisper

## how to finetune

1. Install libraries,

```bash
pip3 install torch==2.5.1
pip3 install accelerate==1.1.1 transformers==4.47.0
pip3 install git+https://github.com/mesolitica/ml-cross-entropy-whisper
```

2. Train,

```bash
bash run-small.sh
```