WANDB_PROJECT="dia-tts-malaysian-emilia-full-mixed-precision-multipacking" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port 29502 \
-m dia_multipacking \
--per_device_train_batch_size 10 \
--gradient_accumulation_steps 1 \
--output_dir dia-tts-malaysian-emilia-full-mixed-precision-multipacking \
--bf16 --do_train --do_eval false --num_train_epochs 2 \
--train_file 'mesolitica/Malaysian-Emilia-Sesame' \
--merged_file 'merged-dia-4096.json' \
--logging_steps 2 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--warmup_steps 100 \
--save_steps 1000 \
--gradient_checkpointing true \
--ddp_find_unused_parameters false \
--dataloader_pin_memory false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 10 \
--remove_unused_columns false