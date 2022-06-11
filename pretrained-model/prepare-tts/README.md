# Prepare TTS

Prepare TTS dataset for TTS models.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

## how-to

1. True case texts for Male and Female speakers,

Female speaker, [true-case-female.ipynb](true-case-female.ipynb), [true-case-female.json](https://f000.backblazeb2.com/file/malaya-speech-model/dataset/true-case-female.json).

Male speaker, [true-case-male.ipynb](true-case-male.ipynb), [true-case-male.json](https://f000.backblazeb2.com/file/malaya-speech-model/dataset/true-case-male.json).

2. Download and preprocess dataset, 

Female speaker, [preprocess-female.ipynb](preprocess-female.ipynb).

Male speaker, [preprocess-female.ipynb](preprocess-male.ipynb).

Husein speaker, [preprocess-female.ipynb](preprocess-husein.ipynb).

Haqkiem speaker, [preprocess-haqkiem.ipynb](preprocess-haqkiem.ipynb).

3. Split dataset to train and test set, 

Female speaker, [train-test-tts-female.ipynb](train-test-tts-female.ipynb), [mels-female.json](https://f000.backblazeb2.com/file/malaya-speech-model/dataset/mels-female.json).

Male speaker, [train-test-tts-male.ipynb](train-test-tts-male.ipynb), [mels-male.json](https://f000.backblazeb2.com/file/malaya-speech-model/dataset/mels-male.json).