WANDB_PROJECT="malay-orpheus-3b-0.1-ft-lora-128" \
HF_HOME="/workspace/cache" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0,1,2,3" \
torchrun --nproc_per_node 4 \
-m train_multipacking \
--model_name_or_path canopylabs/orpheus-3b-0.1-ft \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 6 \
--output_dir lora-128-orpheus-3b-0.1-ft \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file 'packing-3k' \
--logging_steps 1 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--save_steps 100 \
--save_total_limit 3 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--rank 128 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--torch_compile