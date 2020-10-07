# Speaker Vector

Inception V4 trained on Voxceleb.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## How-to

1. Do voxceleb processing, look at [../voxceleb](../voxceleb).

2. Train [inception-v4-softmax.py](inception-v4-softmax.py).

This will train inception v4 from scratch to classify more than 5k unique speakers.

3. Load generated Tensorflow checkpoints and run prediction, [inception-v4-tf.ipynb](inception-v4-tf.ipynb).

## Download

1. Last update, 4th October 2020, [inception-v4-voxceleb-04-10-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/inception-v4-voxceleb-04-10-2020.tar.gz)

  - 410k steps.
  - accuracy = 0.9166667, global_step = 401000, loss = 0.33683872
  - Tensorboard, https://tensorboard.dev/experiment/4qWmM4JeTeerdJXkrHViBg/