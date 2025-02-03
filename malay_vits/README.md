# malay-VITS

Originally from https://github.com/jaywalnut310/vits, https://arxiv.org/abs/2106.06103

## Preprocessing

1. Simply run,

```bash
python3 preprocessing.py --file husein-00000-of-00001.parquet
```

Make sure prepare a parquet file like https://huggingface.co/datasets/mesolitica/TTS

## how-to train

1. Build Monotonic Alignment Search,

```bash
cd monotonic_align
python3 setup.py build_ext --inplace
```

### Single speaker

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=vits-husein-v2 \
python3.10 train.py -c config/husein.json -m husein-v2
```

### Multi speakers

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=vits-multispeaker-medium \
python3.10 train_ms.py -c config/multispeaker-clean-medium.json -m multispeaker-medium
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT=vits-multispeaker-v3 \
python3.10 train_ms.py -c config/multispeaker-clean-v2.json -m multispeaker-v4
```

## Citation

```bibtex
@misc{kim2021conditional,
      title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech}, 
      author={Jaehyeon Kim and Jungil Kong and Juhee Son},
      year={2021},
      eprint={2106.06103},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```