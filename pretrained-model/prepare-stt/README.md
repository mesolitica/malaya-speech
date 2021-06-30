# Prepare ASR

Prepare ASR dataset for ASR models.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

## how-to

1. Download strong semisupervised audiobook and augment it, [prepare-audiobook-augmentation.ipynb](prepare-audiobook-augmentation.ipynb).

2. Run [prepare-malay-stt-train.ipynb](prepare-malay-stt-train.ipynb) to download train Malay dataset.

3. Run [prepare-malay-stt-test.ipynb](prepare-malay-stt-test.ipynb) to download test Malay dataset.

4. Run [prepare-mixed-stt-train.ipynb](prepare-mixed-stt-train.ipynb) to download train Mixed (Malay and Singlish) dataset.

5. Run [prepare-mixed-stt-test.ipynb](prepare-mixed-stt-test.ipynb) to download test Mixed (Malay and Singlish) dataset.

## Download

1. Test set, audio files, [bahasa-asr-testset.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/bahasa-asr-testset.tar.gz), transcript, [bahasa-asr-testset.json](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/bahasa-asr-testset.json).