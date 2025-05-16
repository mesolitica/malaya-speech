WANDB_PROJECT="sesame-1b-malaysian-emilia-full-mixed-precision" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port 29501 \
-m train \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 1 \
--output_dir sesame-1b-malaysian-emilia-full-mixed-precision \
--bf16 --do_train --do_eval false --num_train_epochs 2 \
--train_file 'mesolitica/Malaysian-Emilia-Sesame' \
--merged_file 'merged.json' \
--logging_steps 2 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--warmup_steps 100 \
--save_steps 1500 \
--gradient_checkpointing true \
--ddp_find_unused_parameters false \
--dataloader_pin_memory true \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 10 \
--calculated_speech_tokens true \
--max_length 1536