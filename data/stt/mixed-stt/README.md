# Mixed-STT

Malay, Singlish and Mandarin STT dataset. Scripts how to gather dataset at https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt

Train test split at [huggingface-3mixed-train-test.json](huggingface-3mixed-train-test.json).

1. Malay,

- mean length, 5.8575222180546636 seconds
- minimum length, 0.3453125 seconds
- maximum length, 18 seconds.
- percentile length, [10, 25, 50, 75, 90] = [4.056  , 5.07   , 5.36825, 6.12275, 8.04   ]
- total length, 2398 hours.

calculated at [calculate-distribution-malay.ipynb](calculate-distribution-malay.ipynb).

2. Singlish,
3. Mandarin, 

## Download

All tfrecords available at https://huggingface.co/huseinzol05/STT-Mixed-TFRecord

## how-to

Simple torch dataset, https://github.com/huseinzol05/malaya-speech/blob/master/pretrained-model/stt/hf-wav2vec2/remote-tfrecord.py