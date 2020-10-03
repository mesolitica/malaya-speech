# Speaker Vector

Inception V4 trained on Voxceleb.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## How-to

1. Do voxceleb processing, look at [../voxceleb](../voxceleb).

2. Train [inception-v4-softmax.py](inception-v4-softmax.py).

This will train inception v4 from scratch to classify more than 5k unique speakers.

3. Load generated Tensorflow checkpoints and run prediction, [inception-v4-tf.ipynb](inception-v4-tf.ipynb).

## Download

1. https://f000.backblazeb2.com/file/malaya-speech-model/pretrained/inception-v4-30-09-2020.tar.gz

```
INFO:tensorflow:train_accuracy = 0.8714181, train_loss = 1.0315295 (0.912 sec)
INFO:tensorflow:loss = 0.5727377, step = 257612 (0.912 sec)
INFO:tensorflow:global_step/sec: 1.03318
INFO:tensorflow:train_accuracy = 0.87141764, train_loss = 0.6480751 (0.968 sec)
INFO:tensorflow:loss = 0.39287624, step = 257613 (0.968 se
```