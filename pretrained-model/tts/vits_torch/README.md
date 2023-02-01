# VITS-Torch

Originally from https://github.com/malaysia-ai/projects, code been changed to savely train using malaya-speech characters and methods.

## how-to

1. git clone,

```bash
git clone https://github.com/malaysia-ai/projects
cd projects/malaysia_ai_projects/malay_vits/
```

2. build monothonic alignment,

```bash
cd monotonic_align
python3 setup.py build_ext --inplace
cd ../
```

3. train,

```bash
python3 train.py -c female_singlish.json -m female_singlish
python3 train.py -c haqkiem.json -m haqkiem
python3 train.py -c osman.json -m osman
python3 train.py -c yasmin.json -m yasmin
python3 train.py -c yasmin.json -m yasmin
python3 train.py -c ms-MY-Wavenet-A.json -m ms-MY-Wavenet-A
python3 train.py -c ms-MY-Wavenet-B.json -m ms-MY-Wavenet-B
```

## download

1. Female singlish speaker, https://huggingface.co/mesolitica/VITS-female-singlish

  - Lower case, understand `.,?!` punctuations.

2. Haqkiem speaker, https://huggingface.co/mesolitica/VITS-haqkiem

  - Lower case, understand `.,?!` punctuations.

3. Yasmin speaker, https://huggingface.co/mesolitica/VITS-yasmin

  - Case sensitive, understand `.,?!` punctuations.

4. Osman speaker, https://huggingface.co/mesolitica/VITS-osman

  - Case sensitive, understand `.,?!` punctuations.