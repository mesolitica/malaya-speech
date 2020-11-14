# Prepare Language Model

Prepare language model using KenLM library for CTC decoder.

## Build KenLM

```bash
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```

## how-to

**Make sure you already build KenLM**.

1. Malaya-speech transcript,

```bash
wget https://f000.backblazeb2.com/file/malaya-speech-model/collections/malaya-speech-transcript.txt
kenlm/build/bin/lmplz --text malaya-speech-transcript.txt --arpa out.arpa -o 3 --prune 0 1 1
kenlm/build/bin/build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm
```

2. local language,

Download and prepare dataset, [prepare-local-lm.ipynb](prepare-local-lm.ipynb).

```bash
wget https://f000.backblazeb2.com/file/malaya-speech-model/v1/vocab/cleaned-local.txt
kenlm/build/bin/lmplz --text cleaned-local.txt --arpa local.arpa -o 5 --prune 0 1 1 1 1
kenlm/build/bin/build_binary -q 8 -b 7 -a 256 trie local.arpa local.trie.klm
```