WANDB_PROJECT=vq-bigvgan-65k-47tps-nogan \
CUDA_VISIBLE_DEVICES="0" \
python3.10 -m torch.distributed.run --nproc_per_node 1 --master-port 29501 \
train_65k_nogan_47tps.py