WANDB_PROJECT="Malaysian-TTS-0.6B-v1" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 qwen3_multipacking_distilcodec_flash.py \
--model_name_or_path "mesolitica/Malaysian-TTS-0.6B-v0.1" \
--per_device_train_batch_size 24 \
--gradient_accumulation_steps 1 \
--output_dir Malaysian-TTS-0.6B-v1 \
--bf16 --do_train --do_eval false --num_train_epochs 20 \
--train_file "tokenized-4k-qwen3-stage2-v2/tokenized-0" \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 0 \
--block_size 4096 \
--save_steps 50 \
--save_total_limit 3 \
--gradient_checkpointing true \
--torch_dtype bfloat16 \
--neftune_noise_alpha 5.0 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 5 \
--remove_unused_columns false