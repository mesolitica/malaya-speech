# malay-VITS

Originally from https://github.com/jaywalnut310/vits, https://arxiv.org/abs/2106.06103

V2 from https://github.com/p0p4k/vits2_pytorch

## how-to train

1. Build Monotonic Alignment Search,

```bash
cd monotonic_align
python3 setup.py build_ext --inplace
```

### VITS 1

```bash
python3 train.py -c config.json -m speaker
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