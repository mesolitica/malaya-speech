WANDB_PROJECT="dia-tts-malaysian-emilia-full-mixed-precision-checkpointing" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" \
torchrun --nproc_per_node 6 --master_port 29502 \
-m dia \
--per_device_train_batch_size 10 \
--gradient_accumulation_steps 1 \
--output_dir dia-tts-malaysian-emilia-full-mixed-precision-checkpointing \
--bf16 --do_train --do_eval false --num_train_epochs 2 \
--train_file 'mesolitica/Malaysian-Emilia-Sesame' \
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
--calculated_speech_tokens true \
--remove_unused_columns false