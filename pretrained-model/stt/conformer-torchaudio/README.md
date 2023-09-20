## how-to

### Conformer base

```bash
CUDA_VISIBLE_DEVICES='1' WANDB_DISABLED=true \
python3 trainer.py \
--model conformer_rnnt_base \
--train_dataset /home/husein/speech-bahasa/malay-asr-train-shuffled.json \
--val_dataset /home/husein/ssd1/speech-bahasa/malay-asr-test.json \
--output_dir conformer_rnnt_base \
--save_total_limit 3 \
--save_steps 10000 \
--logging_steps 100 \
--warmup_steps 1000 \
--num_train_epochs 2
```