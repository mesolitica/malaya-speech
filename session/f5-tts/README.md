# F5-TTS

## how to finetune

1. Download dataset,

```bash
bash download.sh
```

2. Install libraries,

```bash
cd /workspace
git clone https://github.com/mesolitica/F5-TTS
cd F5-TTS
git submodule update --init --recursive
pip3 install -e .
pip3 install torchdiffeq x-transformers jieba pypinyin ema_pytorch accelerate==1.1.1 torch==2.5.1 torchaudio==2.5.1
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Voice-Conversion', repo_type='dataset', allow_patterns = 'data/Emilia_Malaysian_pinyin/*', local_dir = './')
"
mkdir ckpts/F5TTS_Base

# vocos
wget https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt -O ckpts/F5TTS_Base/model_1200000.pt

# bigvgan
wget https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base_bigvgan/model_1250000.pt -O ckpts/F5TTS_Base/model_1250000.pt
```

3. Train,

```bash
cd /workspace/F5-TTS
accelerate config
accelerate launch src/f5_tts/train/train.py
```