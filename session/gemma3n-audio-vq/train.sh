WANDB_PROJECT="gemma3n-audio-vq" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
torchrun --nproc_per_node 1 \
-m finetune_vq \
--model_name_or_path "huseinzol05/gemma-3n-e4b-it-audio-encoder" \
--per_device_train_batch_size 128 \
--gradient_accumulation_steps 1 \
--output_dir gemma3n-audio-vq \
--bf16 --do_train --do_eval false \
--num_train_epochs 5 \
--train_file filtered-train-cv-17.parquet \
--logging_steps 1 \
--learning_rate 1e-4 \
--lr_scheduler_type linear \
--warmup_steps 100 \
--weight_decay 0.0 \
--save_steps 100 \
--save_total_limit 5 \
--gradient_checkpointing false \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 5 \
--remove_unused_columns false