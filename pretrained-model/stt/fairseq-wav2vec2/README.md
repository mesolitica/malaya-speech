```bash
fairseq-hydra-train \
task.data=/home/husein/ssd1/speech-bahasa/wav2vec2-malay \
--config-dir /home/husein/fairseq/examples/wav2vec/config/pretraining \
--config-name wav2vec2_base_librispeech
```

```bash
fairseq-hydra-train \
task.data=/home/husein/ssd1/speech-bahasa/wav2vec2-malay \
--config-dir /home/husein/fairseq/examples/wav2vec/config/pretraining \
--config-name small
```

```bash
CUDA_VISIBLE_DEVICES=0 \
fairseq-hydra-train \
task.data=/home/husein/wav2vec2-malay-singlish \
--config-dir /home/husein/fairseq/examples/wav2vec/config/pretraining \
--config-name wav2vec2_base_librispeech
```

```bash
CUDA_VISIBLE_DEVICES=0 \
fairseq-hydra-train \
task.data=/home/husein/wav2vec2-malay-singlish \
--config-dir /home/husein/fairseq/examples/wav2vec/config/pretraining \
--config-name wav2vec2_base_librispeech
```

```bash
CUDA_VISIBLE_DEVICES=1 \
fairseq-hydra-train \
task.data=/home/husein/wav2vec2-malay-singlish \
--config-dir /home/husein/fairseq/examples/wav2vec/config/pretraining \
--config-name small
```

```bash
CUDA_VISIBLE_DEVICES=0 \
fairseq-hydra-train \
task.data=/home/husein/wav2vec2-malay-singlish \
distributed_training.distributed_world_size=1 \
optimization.update_freq='[1]' \
--config-dir /home/husein/fairseq/examples/wav2vec/config/pretraining \
--config-name semi-large
```