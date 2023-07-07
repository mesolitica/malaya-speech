# Prepare ASR

Prepare ASR dataset for ASR models.

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

## how-to

1. Download strong semisupervised audiobook and augment it, [prepare-audiobook-augmentation.ipynb](prepare-audiobook-augmentation.ipynb).

2. Run [prepare-malay-stt-train.ipynb](prepare-malay-stt-train.ipynb) to download train Malay dataset.

3. Run [prepare-malay-stt-test.ipynb](prepare-malay-stt-test.ipynb) to download test Malay dataset.

4. Run [prepare-mixed-stt-train.ipynb](prepare-mixed-stt-train.ipynb) to download train Mixed (Malay and Singlish) dataset.

5. Run [prepare-mixed-stt-test.ipynb](prepare-mixed-stt-test.ipynb) to download test Mixed (Malay and Singlish) dataset.

## Download

1. Malay Test set,

- ~53 minutes.
- audio files, https://huggingface.co/datasets/mesolitica/stt-dataset/resolve/main/malay-test.tar.gz
- transcription, [postprocess-malaya-malay-test-set.json](postprocess-malaya-malay-test-set.json)

2. Singlish Test set,

- ~271 minutes.
- audio files, https://huggingface.co/datasets/mesolitica/stt-dataset/resolve/main/singlish-test.tar.gz
- transcription, [postprocess-malaya-malay-test-set.json](postprocess-malaya-malay-test-set.json)

3. Mandarin Test set,

- ~46 minutes.
- audio files, https://huggingface.co/datasets/mesolitica/stt-dataset/resolve/main/mandarin-test.tar.gz
- transcription, [mandarin-test.json](https://f000.backblazeb2.com/file/malaya-speech-model/asr-dataset/mandarin-test.json)

4. Processed FLEURS102 `ms_my`,

- ~133 minutes.
- audio files, https://huggingface.co/datasets/mesolitica/stt-dataset/resolve/main/malay-fleur102.tar.gz
- transcription, [malay-asr-test.json](malay-asr-test.json).

5. Malay train dataset,

- notebook, [prepare-malay-stt-train.ipynb](prepare-malay-stt-train.ipynb)
- transcription, https://huggingface.co/datasets/mesolitica/stt-dataset/resolve/main/malay-asr-train.json

6. Malay train dataset + Semisupervised Large Conformer,

- notebook, [gather-semisupervised-asr.ipynb](gather-semisupervised-asr.ipynb).
- transcription, https://huggingface.co/datasets/mesolitica/stt-dataset/resolve/main/malay-asr-train-shuffled-combined-semi.json

7. Mixed train dataset,

- notebook, [prepare-stt-mixed-v2.ipynb](prepare-stt-mixed-v2.ipynb).
- transcription, https://huggingface.co/datasets/mesolitica/stt-dataset/resolve/main/mixed-stt-train-v2.json