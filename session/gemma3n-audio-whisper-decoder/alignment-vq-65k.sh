WANDB_PROJECT=gemma3n-audio-vq-whisper-decoder-65k \
CUDA_VISIBLE_DEVICES="0" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
torchrun --nproc_per_node 1 --master-port 29501 \
-m gemmavqwhisper_65k \
--model_name_or_path "mesolitica/gemma3n-audio-encoder-whisper-decoder" \
--train_dataset_name "mosaic-stt-include-malaysian" \
--eval_steps 1000 \
--save_steps 500 \
--warmup_steps 100 \
--learning_rate 0.00002 \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 3 \
--per_device_train_batch_size 80 \
--gradient_accumulation_steps 1 \
--per_device_eval_batch_size 1 \
--output_dir "gemma3n-audio-vq-whisper-decoder-64k" \
--do_train \
--max_label_length 400 \
--bf16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 32 \
--dataloader_prefetch_factor 2