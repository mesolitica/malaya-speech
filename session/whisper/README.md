# Finetune Whisper

## how-to

### Small

```bash
~/.local/bin/torchrun --nproc_per_node 1 \
-m finetune \
--model_name_or_path "openai/whisper-small" \
--train_dataset_name "sample-set.jsonl" \
--eval_dataset_name "sample-set.jsonl" \
--eval_steps 1000 \
--save_steps 100 \
--warmup_steps 50 \
--learning_rate 0.0001 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--output_dir "./small" \
--do_train \
--do_eval \
--gradient_checkpointing \
--predict_with_generate \
--overwrite_output_dir \
--max_label_length 450 \
--bf16
```