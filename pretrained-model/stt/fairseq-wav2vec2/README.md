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