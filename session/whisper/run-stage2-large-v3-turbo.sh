WANDB_PROJECT=run-stage2-large-v3-turbo \
CUDA_VISIBLE_DEVICES="0,1" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
torchrun --nproc_per_node 2 \
-m whisper_v2 \
--model_name_or_path "openai/whisper-large-v3-turbo" \
--train_dataset_name "mosaic-stt-post" \
--eval_steps 1000 \
--save_steps 300 \
--warmup_steps 100 \
--learning_rate 0.00005 \
--logging_steps 2 \
--save_total_limit 3 \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--per_device_eval_batch_size 4 \
--output_dir "run-stage2-large-v3-turbo" \
--gradient_checkpointing \
--do_train \
--ddp_find_unused_parameters false \
--max_label_length 440 \
--bf16 \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--neftune_noise_alpha 5.0 \
--torch_compile