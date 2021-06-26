# Prepare Language Model

Prepare language model using KenLM library for CTC decoder.

## how-to

1. Build KenLM,

```bash
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```

2. Run KenLM,

```bash
kenlm/build/bin/lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1
kenlm/build/bin/build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm
```