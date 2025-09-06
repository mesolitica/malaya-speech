export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="Qwen-Qwen3-4B-Base-4k-TTS-distilcodec" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 qwen3_multipacking_distilcodec_flash.py \
--model_name_or_path "Qwen/Qwen3-4B-Base" \
--per_device_train_batch_size 3 \
--gradient_accumulation_steps 4 \
--output_dir Qwen-Qwen3-4B-Base-4k-TTS-distilcodec \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file "packing-qwen3-combine" \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 20 \
--block_size 4096 \
--save_steps 100 \
--save_total_limit 5 \
--gradient_checkpointing true \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 5 \
--remove_unused_columns false