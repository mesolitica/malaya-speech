pip3 install torchaudio==2.7.0 datasets evaluate accelerate transformers==4.53.1
pip3 install librosa soundfile multiprocess
pip3 install mosaicml-streaming networkx==3.2.1
pip3 install numpy==1.26.4
pip3 install tf-keras==2.16.0 --no-dependencies
pip3 install pip -U
pip3 install git+https://github.com/apple/ml-cross-entropy
pip3 install wandb

huggingface-cli download --repo-type dataset \
--local-dir './' \
huseinzol05/mosaic-STT-VQ

huggingface-cli download --repo-type dataset \
--include 'Malaysian-Multiturn-Chat-Assistant-manglish-*.zip' \
--local-dir './' \
mesolitica/Malaysian-Multiturn-Chat-Assistant

huggingface-cli download --repo-type dataset \
--include 'Malaysian-Multiturn-Chat-Assistant-*.zip' \
--local-dir './' \
mesolitica/Malaysian-Multiturn-Chat-Assistant

huggingface-cli download --repo-type dataset \
--include '*.zip' \
--local-dir './' \
malaysia-ai/common_voice_17_0

wget https://gist.githubusercontent.com/huseinzol05/2e26de4f3b29d99e993b349864ab6c10/raw/9b2251f3ff958770215d70c8d82d311f82791b78/unzip.py
python3 unzip.py