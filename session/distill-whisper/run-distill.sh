torchrun --nproc_per_node 4 \
-m distill \
--model_name_or_path "./distil-large-v3-init" \
--teacher_model_name_or_path "openai/whisper-large-v3" \
--train_dataset_name "mosaic-combine-stt" \
--eval_dataset_name "test-fleurs.json" \
--eval_steps 100 \
--save_steps 100 \
--warmup_steps 100 \
--learning_rate 0.00005 \
--dtype "bfloat16" \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 3 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 4 \
--output_dir "./distil-large-v3" \
--do_train \
--gradient_checkpointing \
--predict_with_generate \
--freeze_encoder \
--max_label_length 450