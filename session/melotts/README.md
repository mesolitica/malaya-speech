# MeloTTS

## how to train

1. Download dataset,

```bash
bash download.sh
```

2. Prepare dataset,

```bash
cd /workspace
git clone https://github.com/mesolitica/MeloTTS-MS
cd MeloTTS-MS
git checkout char
pip3 install -e .
pip3 install git+https://github.com/mesolitica/malaya-speech malaya mecab-python3 transformers==4.47.0 PySastrawi matplotlib==3.7.0 torch==2.5.1
python3 -c "
import nltk
nltk.download('averaged_perceptron_tagger_eng')
"
cd melo
python3 preprocess_text.py --metadata /workspace/melotts.txt --num_device 1 --max-val-total 12
bash train.sh /workspace/config.json 1
```