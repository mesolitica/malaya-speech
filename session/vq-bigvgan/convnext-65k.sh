WANDB_PROJECT=convnext-65k \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="0" \
python3 -m torch.distributed.run --nproc_per_node 1 --master-port 29501 \
convnext_65k.py