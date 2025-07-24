WANDB_PROJECT="llama3-tts-100M-nonpack" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
torchrun --nproc_per_node 1 \
-m train_multipacking \
--model_name_or_path "./100M-v3" \
--per_device_train_batch_size 80 \
--gradient_accumulation_steps 1 \
--output_dir llama3-tts-100M-nonpack \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file "postfilter-idayu.parquet,postfilter-husein.parquet" \
--mosaic false \
--logging_steps 1 \
--learning_rate 1e-4 \
--lr_scheduler_type linear \
--warmup_steps 1000 \
--block_size 1290 \
--save_steps 2000 \
--save_total_limit 5 \
--gradient_checkpointing true \
--max_grad_norm 5.0 \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 3 \
--dataloader_prefetch_factor 3 \
--torch_compile