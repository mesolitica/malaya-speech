# StyleTTS2

## how to train

1. Download dataset,

```bash
bash download.sh
```

2. Prepare dataset,

```bash
cd /ephemeral
git clone https://github.com/mesolitica/StyleTTS2-MS
cd StyleTTS2-MS
pip3 install -r requirements.txt
pip3 install git+https://github.com/mesolitica/malaya-speech malaya phonemizer mecab-python3 transformers==4.47.0 PySastrawi matplotlib==3.7.0 torch==2.5.1 torchaudio==2.5.1 accelerate==1.1.1 munch einops einops_exts pandas tensorboard
apt update
apt install espeak espeak-ng -y
python3 train_first.py --config_path Configs/config_multispeakers.yml
```