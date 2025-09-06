WANDB_PROJECT=vq-bigvgan-65k-nogan-v2 \
CUDA_VISIBLE_DEVICES="1" \
python3.10 -m torch.distributed.run --nproc_per_node 1 --master-port 29502 \
train_65k_nogan.py