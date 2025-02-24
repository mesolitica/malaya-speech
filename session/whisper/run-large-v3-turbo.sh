WANDB_PROJECT=malaysian-whisper-large-v3-turbo-v2 \
torchrun --nproc_per_node 2 \
-m whisper \
--model_name_or_path "openai/whisper-large-v3-turbo" \
--train_dataset_name "mosaic-stt" \
--eval_steps 1000 \
--save_steps 100 \
--warmup_steps 100 \
--learning_rate 0.00005 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 3 \
--per_device_train_batch_size 24 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 4 \
--output_dir "malaysian-whisper-large-v3-turbo-v2" \
--do_train \
--gradient_checkpointing \
--predict_with_generate \
--max_label_length 440 \
--bf16 \
--dataloader_num_workers 3 \
--dataloader_prefetch_factor 4