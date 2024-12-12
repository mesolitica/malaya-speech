# F5-TTS

## how to Speech Enhancement

1. Download dataset,

```bash
cd /workspace
wget https://www.7-zip.org/a/7z2301-linux-x64.tar.xz
tar -xf 7z2301-linux-x64.tar.xz
pip3 install huggingface-hub wandb
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'filtered-24k_processed.z*', local_dir = './')
"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'malaysian-podcast-processed.z*', local_dir = './')
"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'sg-podcast_processed.zip', local_dir = './')
"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'malaysian-cartoon.z*', local_dir = './')
"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'parlimen-24k-chunk_processed.z*', local_dir = './')
"
/workspace/7zz x filtered-24k_processed.zip -y -mmt40
/workspace/7zz x malaysian-podcast-processed.zip -y -mmt40
/workspace/7zz x sg-podcast_processed.zip -y -mmt40
/workspace/7zz x parlimen-24k-chunk_processed.zip -y -mmt40
/workspace/7zz x malaysian-cartoon.zip -y -mmt40
```

2. Install libraries,

```bash
git clone https://github.com/mesolitica/F5-TTS
cd F5-TTS
pip3 install -e .
pip3 install torchdiffeq x-transformers jieba pypinyin ema_pytorch accelerate==1.1.1 torch==2.5.1 torchaudio==2.5.1
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Voice-Conversion', repo_type='dataset', allow_patterns = 'data/Emilia_Malaysian_pinyin/*', local_dir = './')
"
mkdir ckpts/F5TTS_Base
wget https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt -O ckpts/F5TTS_Base/model_1200000.pt
```

3. Train,

```bash
apt update
apt install screen vim -y
accelerate config
accelerate launch src/f5_tts/train/train.py
```