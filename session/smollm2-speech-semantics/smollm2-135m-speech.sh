WANDB_PROJECT="run-instruction-speech-multipack-HuggingFaceTB-SmolLM2-135M" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 2 \
-m run-instruction-speech-multipack \
--model_name_or_path HuggingFaceTB/SmolLM2-135M \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 2 \
--output_dir run-instruction-speech-multipack-HuggingFaceTB-SmolLM2-135M \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file /home/husein/ssd4/continue-training/smollm2-speech-semantic-multipack-2048 \
--logging_steps 1 \
--learning_rate 5e-4 \
--warmup_steps 200 \
--block_size 24576 \
--save_steps 1000 \
--save_total_limit 3 \
--gradient_checkpointing true \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 3 \
--dataloader_prefetch_factor 4 \
--torch_compile \
--torch_compile_backend inductor