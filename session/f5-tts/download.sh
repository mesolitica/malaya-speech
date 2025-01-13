#!/bin/bash

apt update
apt install unzip ffmpeg -y
cd /workspace
wget https://www.7-zip.org/a/7z2301-linux-x64.tar.xz
tar -xf 7z2301-linux-x64.tar.xz
pip3 install huggingface-hub wandb

cmd1="
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'filtered-24k_processed.z*', local_dir = './')
\"
/workspace/7zz x filtered-24k_processed.zip -y -mmt40
rm filtered-24k_processed.z*
"

cmd2="
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'malaysian-podcast-processed.z*', local_dir = './')
\"
/workspace/7zz x malaysian-podcast-processed.zip -y -mmt40
rm malaysian-podcast-processed.zip
"

cmd3="
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'sg-podcast_processed.zip', local_dir = './')
\"
/workspace/7zz x sg-podcast_processed.zip -y -mmt40
rm sg-podcast_processed.zip
"

cmd4="
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'malaysian-cartoon.z*', local_dir = './')
\"
/workspace/7zz x malaysian-cartoon.zip -y -mmt40
rm malaysian-cartoon.z*
"

cmd5="
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'parlimen-24k-chunk_processed.z*', local_dir = './')
\"
/workspace/7zz x parlimen-24k-chunk_processed.zip -y -mmt40
rm parlimen-24k-chunk_processed.z*
"

cmd6="
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Emilia', repo_type='dataset', allow_patterns = 'dialects-processed-*.zip', local_dir = './', max_workers = 20)
\"
wget https://raw.githubusercontent.com/mesolitica/malaysian-dataset/refs/heads/master/text-to-speech/emilia/unzip.py
python3 unzip.py
"

cmd7="
cd /workspace
wget https://huggingface.co/datasets/mesolitica/Malaysian-Emilia/resolve/main/klasik_processed.zip
unzip klasik_processed.zip
rm klasik_processed.zip

wget https://huggingface.co/datasets/mesolitica/Malaysian-Voice-Conversion/resolve/main/data/classic-malay-chunk.zip
unzip classic-malay-chunk.zip
rm classic-malay-chunk.zip

wget https://huggingface.co/datasets/mesolitica/Malaysian-Voice-Conversion/resolve/main/data/parliament-chunk.zip
unzip parliament-chunk.zip
rm parliament-chunk.zip

wget https://huggingface.co/datasets/mesolitica/Malaysian-Voice-Conversion/resolve/main/data/text-chunk-podcasts.zip
unzip text-chunk-podcasts.zip
rm text-chunk-podcasts.zip
"

cmd8="
cd /workspace
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mesolitica/Malaysian-Voice-Conversion', repo_type='dataset', allow_patterns = 'data/other-chunk.z*', local_dir = './')
\"
mv data/other-chunk.z* .
/workspace/7zz x other-chunk.zip -y -mmt40
rm other-chunk.z*
"

cmd9="
cd /workspace
wget https://huggingface.co/datasets/mesolitica/Extra-Emilia/resolve/main/emilia-mandarin.zip
unzip emilia-mandarin.zip
rm emilia-mandarin.zip
"

bash -c "$cmd1" &
pid1=$!

bash -c "$cmd2" &
pid2=$!

bash -c "$cmd3" &
pid3=$!

bash -c "$cmd4" &
pid4=$!

bash -c "$cmd5" &
pid5=$!

bash -c "$cmd6" &
pid6=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6

echo "All processes completed!"

bash -c "$cmd7" &
pid7=$!

bash -c "$cmd8" &
pid8=$!

bash -c "$cmd9" &
pid9=$!

wait $pid7 $pid8 $pid9

echo "All processes completed!"