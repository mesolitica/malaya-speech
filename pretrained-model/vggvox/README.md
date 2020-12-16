# Speaker Vector

Trained on Voxceleb. Provided 2 versions, 

1. V1, https://github.com/linhdvu14/vggvox-speaker-identification
2. V2, https://github.com/WeidiXie/VGG-Speaker-Recognition

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator, Tensorflow Dataset is really helpful**.

## How-to

1. Change from Keras checkpoint to Tensorflow, run [v1-keras-to-tf.ipynb](v1-keras-to-tf.ipynb) or [v2-keras-to-tf.ipynb](v1-keras-to-tf.ipynb).

2. Load generated Tensorflow checkpoints and run prediction, [v1-tf.ipynb](v1-tf.ipynb) or [v2-tf.ipynb](v2-tf.ipynb).

3. Freeze the models for production, [export-v1.ipynb](export-v1.ipynb) or [export-v2.ipynb](export-v2.ipynb).

## Download

**These checkpoints are mirrored**.

1. V1, 70.8 MB, https://f000.backblazeb2.com/file/malaya-speech-model/vggvox/weights-v1.h5

2. V2, 31.1 MB, https://f000.backblazeb2.com/file/malaya-speech-model/vggvox/weights-v2.h5