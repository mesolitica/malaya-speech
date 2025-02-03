#!/bin/bash

apt update
apt install unzip ffmpeg -y
apt install screen vim -y
apt update && apt install -y locales
locale-gen en_US.UTF-8
cd /ephemeral
pip3 install huggingface-hub wandb multiprocess notebook==6.5.6
wget https://www.7-zip.org/a/7z2301-linux-x64.tar.xz
tar -xf 7z2301-linux-x64.tar.xz
screen -dmS jupyter_session bash -c "jupyter notebook --NotebookApp.token='' --no-browser --allow-root --notebook-dir='/ephemeral'"

cmd1="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /ephemeral
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/TTS', repo_type='dataset', allow_patterns = 'processed/anwar-ibrahim.z*', local_dir = './')
\"
mv processed/anwar-ibrahim.z* .
/ephemeral/7zz x anwar-ibrahim.zip -y -mmt40
rm anwar-ibrahim.z*
"

cmd2="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /ephemeral
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/TTS', repo_type='dataset', allow_patterns = 'processed/husein.z*', local_dir = './')
\"
mv processed/husein.z* .
/ephemeral/7zz x husein.zip -y -mmt40
rm husein.z*
"

cmd3="
export LC_ALL=en_US.UTF-8; export LANG=en_US.UTF-8;
cd /ephemeral
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/TTS', repo_type='dataset', allow_patterns = 'processed/shafiqah-idayu.z*', local_dir = './')
\"
mv processed/shafiqah-idayu.z* .
/ephemeral/7zz x shafiqah-idayu.zip -y -mmt40
rm shafiqah-idayu.z*
"

cmd4="
cd /ephemeral
wget https://huggingface.co/mesolitica/StyleTTS2-MS/resolve/main/checkpoints-first-stage/epoch_1st_00004.pth
wget https://huggingface.co/datasets/mesolitica/TTS/resolve/main/styletts2/train_list.txt
wget https://huggingface.co/datasets/mesolitica/TTS/resolve/main/styletts2/val_list.txt
"

bash -c "$cmd1" &
pid1=$!

bash -c "$cmd2" &
pid2=$!

bash -c "$cmd3" &
pid3=$!

bash -c "$cmd4" &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4

echo "All processes completed!"