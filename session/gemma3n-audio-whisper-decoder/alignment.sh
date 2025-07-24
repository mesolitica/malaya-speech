WANDB_PROJECT=gemma3n-audio-whisper-decoder \
CUDA_VISIBLE_DEVICES="0,1" \
torchrun --nproc_per_node 2 --master-port 29501 \
-m gemmawhisper \
--model_name_or_path "openai/whisper-large-v3-turbo" \
--train_dataset_name "mosaic-stt" \
--eval_steps 1000 \
--save_steps 500 \
--warmup_steps 100 \
--learning_rate 0.00002 \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 1 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--per_device_eval_batch_size 1 \
--output_dir "gemma3n-audio-whisper-decoder" \
--do_train \
--max_label_length 400 \
--bf16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 5