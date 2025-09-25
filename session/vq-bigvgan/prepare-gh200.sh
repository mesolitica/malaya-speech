pip3 install multiprocess huggingface-hub

huggingface-cli download --repo-type dataset \
--include '*.zip' \
--local-dir './' \
--max-workers 20 \
malaysia-ai/common_voice_17_0

huggingface-cli download --repo-type dataset \
--include '*.zip' \
--local-dir './' \
--max-workers 20 \
huseinzol05/chunk-20s

huggingface-cli download mesolitica/VQ-65k-BigVGAN-47TPS audio-files.json \
--local-dir './'

wget https://gist.githubusercontent.com/huseinzol05/2e26de4f3b29d99e993b349864ab6c10/raw/9b2251f3ff958770215d70c8d82d311f82791b78/unzip.py
python3 unzip.py

git clone https://github.com/NVIDIA/BigVGAN
pip3 install torchaudio==2.7.0 nnaudio librosa soundfile wandb numpy==1.26.4
