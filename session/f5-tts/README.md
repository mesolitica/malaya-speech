# F5-TTS

## how to finetune

1. Download dataset,

```bash
bash download.sh
```

2. Install libraries,

```bash
cd /workspace
git clone https://github.com/SWivid/F5-TTS
cd F5-TTS
pip3 install -e .
mkdir ckpts/F5TTS_v1_Base_vocos_pinyin_Emilia_Malaysian
wget https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/model_1250000.safetensors -O ckpts/F5TTS_v1_Base_vocos_pinyin_Emilia_Malaysian/pretrained_model_1250000.safetensors
```

3. Train,

```bash
accelerate config
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py \
--config-name F5TTS_v1_Base.yaml \
++optim.num_warmup_updates=5000 \
++optim.learning_rate=2.5e-5 \
++datasets.name=Emilia_Malaysian \
++datasets.batch_size_per_gpu=42400 \
++model.arch.checkpoint_activations=True \
++ckpts.save_per_updates=10000 \
++ckpts.last_per_updates=5000
```