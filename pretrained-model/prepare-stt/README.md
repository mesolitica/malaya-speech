# Prepare ASR

Prepare ASR dataset for ASR models.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

## how-to

1. Download strong semisupervised audiobook and augment it, [prepare-audiobook-augmentation.ipynb](prepare-audiobook-augmentation.ipynb).

2. Run [download-and-prepare-malaya-speech-train-dataset-v2.ipynb](download-and-prepare-malaya-speech-train-dataset-v2.ipynb) to download train dataset, preprocessing and convert to tfrecord.

3. Run [download-and-prepare-malaya-speech-test-dataset-v2.ipynb](download-and-prepare-malaya-speech-test-dataset-v2.ipynb) to download test dataset, preprocessing and convert to tfrecord.

## Download

1. Train set strong semisupervised audiobook, https://f000.backblazeb2.com/file/malaya-speech-model/data/trainset-audiobook.tar.gz
2. cleaned transcript, https://f000.backblazeb2.com/file/malaya-speech-model/collections/malaya-speech-transcript.txt