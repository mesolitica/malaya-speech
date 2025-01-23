# malay-VITS

Originally from https://github.com/jaywalnut310/vits, https://arxiv.org/abs/2106.06103

V2 from https://github.com/p0p4k/vits2_pytorch

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

### VITS 1

#### Single speaker

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=vits-husein python3.10 train.py -c config/husein.json -m husein
```

#### Multi speakers

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=vits-multispeaker-medium python3.10 train_ms.py -c config/multispeaker-clean-medium.json -m multispeaker-v2-medium
```

### VITS 2

```bash
python3 train_v2.py -c config.json -m speaker
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