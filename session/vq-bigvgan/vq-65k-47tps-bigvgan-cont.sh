WANDB_PROJECT=vq-bigvgan-65k-47tps-cont \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="0" \
python3 -m torch.distributed.run --nproc_per_node 1 --master-port 29501 \
train_65k_47tps_cont.py