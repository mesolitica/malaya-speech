# Parler TTS

## dataset

Dataset at https://huggingface.co/datasets/mesolitica/TTS, notebook preparation at [prepare-parlertts.ipynb](prepare-parlertts.ipynb).

## how to finetune

1. Prepare dataset,

```bash
cd /workspace
apt update
apt install ffmpeg zip screen -y
screen -dmS jupyter_session bash -c "jupyter notebook --NotebookApp.token='' --no-browser --allow-root --notebook-dir='/workspace'"
git clone https://github.com/huggingface/parler-tts
cd parler-tts
pip3 install -e .
pip3 install notebook==6.5.6 wandb multiprocess accelerate==1.1.1 datasets evaluate transformers==4.47.0
cd training
wget https://huggingface.co/datasets/huseinzol05/mesolitica-tts-combined/resolve/main/tmp_dataset_audio.zip
wget https://huggingface.co/datasets/huseinzol05/mesolitica-tts-combined/resolve/main/audio_code_tmp.zip
unzip tmp_dataset_audio.zip
unzip audio_code_tmp.zip
```

### Mini V1

```bash
bash parler-mini.sh
```

### Large V1

```bash
bash parler-large.sh
```
