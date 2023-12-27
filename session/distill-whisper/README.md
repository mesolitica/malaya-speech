# Distilling Whisper

## how-to

### Create student model

#### Large

```bash
python3 create_student_model.py \
  --teacher_checkpoint "openai/whisper-large-v3" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./distil-large-v3-init"
```

#### Medium

```bash
python3 create_student_model.py \
  --teacher_checkpoint "openai/whisper-medium" \
  --encoder_layers 24 \
  --decoder_layers 2 \
  --save_dir "./distil-medium-init"
```

#### Small

```bash
python3 create_student_model.py \
  --teacher_checkpoint "openai/whisper-small" \
  --encoder_layers 12 \
  --decoder_layers 2 \
  --save_dir "./distil-small-init"
```

### Distilling

#### Small

```bash
~/.local/bin/torchrun --nproc_per_node 1 \
-m distill \
--model_name_or_path "./distil-small-init" \
--teacher_model_name_or_path "openai/whisper-small" \
--train_dataset_name "sample-set.jsonl" \
--eval_dataset_name "sample-set.jsonl" \
--eval_steps 1000 \
--save_steps 100 \
--warmup_steps 50 \
--learning_rate 0.0001 \
--dtype "bfloat16" \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--output_dir "./distil-small" \
--do_train \
--do_eval \
--gradient_checkpointing \
--predict_with_generate \
--freeze_encoder \
--overwrite_output_dir \
--max_label_length 450
```