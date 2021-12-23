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

1. Malay Test set, audio files, ~53 minutes, [malay-test.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/malay-test.tar.gz), transcript, [malay-test.json](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/malay-test.json).

2. Singlish Test set, audio files, ~271 minutes, [singlish-test.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/singlish-test.tar.gz), transcript, [singlish-test.json](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/singlish-test.json).

3. Mandarin Test set, audio files, ~46 minutes, [mandarin-test.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/mandarin-test.tar.gz), transcript, [mandarin-test.json](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/mandarin-test.json).