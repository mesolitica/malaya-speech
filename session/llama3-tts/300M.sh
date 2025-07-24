WANDB_PROJECT="llama3-tts-300M-4k" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
torchrun --nproc_per_node 1 \
-m train_multipacking \
--model_name_or_path "mesolitica/Malaysian-Llama3-300M-TTS" \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--output_dir llama3-tts-300M-4k-v3 \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file packing-v2 \
--logging_steps 1 \
--learning_rate 1e-4 \
--lr_scheduler_type linear \
--warmup_steps 100 \
--weight_decay 0.01 \
--block_size 4096 \
--save_steps 100 \
--save_total_limit 5 \
--gradient_checkpointing true \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 5 \
--torch_compile