WANDB_PROJECT=malaysian-conformer-ctc-12M \
torchrun --nproc_per_node 2 \
-m train \
--model_name_or_path "huseinzol05/conformer-12M" \
--train_file "/home/husein/ssd3/mosaic-stt" \
--save_steps 1000 \
--warmup_steps 1000 \
--learning_rate 5e-5 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--save_total_limit 3 \
--num_train_epochs 100 \
--per_device_train_batch_size 108 \
--output_dir "./12M" \
--do_train \
--bf16