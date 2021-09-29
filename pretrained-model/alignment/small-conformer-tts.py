import pyroomacoustics as pra
import numpy as np
from pydub import AudioSegment
from sklearn.utils import shuffle
from glob import glob
import random
import json
import malaya_speech.train as train
import malaya_speech.config
import malaya_speech.train.model.transducer as transducer
import malaya_speech.train.model.conformer as conformer
import malaya_speech.augmentation.spectrogram as mask_augmentation
import malaya_speech.augmentation.waveform as augmentation
import malaya_speech
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


subwords = malaya_speech.subword.load('transducer.subword')
config = malaya_speech.config.conformer_small_encoder_config
config['subsampling']['strides'] = 1
sr = 16000
maxlen = 18
minlen_text = 1

parameters = {
    'optimizer_params': {'beta1': 0.9, 'beta2': 0.98, 'epsilon': 10e-9},
    'lr_policy_params': {
        'warmup_steps': 40000,
        'max_lr': (0.05 / config['dmodel']),
    },
}

featurizer = malaya_speech.tf_featurization.STTFeaturizer(
    normalize_per_feature=True
)
n_mels = featurizer.num_feature_bins


def transformer_schedule(step, d_model, warmup_steps=4000, max_lr=None):
    arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
    arg2 = step * (warmup_steps ** -1.5)
    arg1 = tf.cast(arg1, tf.float32)
    arg2 = tf.cast(arg2, tf.float32)
    lr = tf.math.rsqrt(tf.cast(d_model, tf.float32)) * tf.math.minimum(
        arg1, arg2
    )
    if max_lr is not None:
        max_lr = tf.cast(max_lr, tf.float32)
        return tf.math.minimum(max_lr, lr)
    return lr


def learning_rate_scheduler(global_step):

    return transformer_schedule(
        tf.cast(global_step, tf.float32),
        config['dmodel'],
        **parameters['lr_policy_params'],
    )


def mel_augmentation(features):

    features = mask_augmentation.warp_time_pil(features)
    features = mask_augmentation.mask_frequency(features, width_freq_mask=12)
    features = mask_augmentation.mask_time(
        features, width_time_mask=int(features.shape[0] * 0.05)
    )
    return features


def mp3_to_wav(file, sr=sr):
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(sr).set_channels(1)
    sample = np.array(audio.get_array_of_samples())
    return malaya_speech.astype.int_to_float(sample), sr


def generate(file):
    with open(file) as fopen:
        dataset = json.load(fopen)
    audios, cleaned_texts = dataset['X'], dataset['Y']
    while True:
        audios, cleaned_texts = shuffle(audios, cleaned_texts)
        for i in range(len(audios)):
            try:
                if not os.path.exists(audios[i]):
                    continue
                if audios[i].endswith('.mp3'):
                    # print('found mp3', audios[i])
                    wav_data, _ = mp3_to_wav(audios[i])
                else:
                    wav_data, _ = malaya_speech.load(audios[i], sr=sr)

                if (len(wav_data) / sr) > maxlen:
                    # print(f'skipped audio too long {audios[i]}')
                    continue

                if len(cleaned_texts[i]) < minlen_text:
                    # print(f'skipped text too short {audios[i]}')
                    continue

                t = malaya_speech.subword.encode(
                    subwords, cleaned_texts[i], add_blank=False
                )

                back = np.zeros(shape=(2000,))
                front = np.zeros(shape=(200,))
                wav_data = np.concatenate([front, wav_data, back], axis=-1)

                yield {
                    'waveforms': wav_data,
                    'targets': t,
                    'targets_length': [len(t)],
                }
            except Exception as e:
                print(e)


def preprocess_inputs(example):
    s = featurizer.vectorize(example['waveforms'])
    s = tf.reshape(s, (-1, n_mels))
    s = tf.compat.v1.numpy_function(mel_augmentation, [s], tf.float32)
    mel_fbanks = tf.reshape(s, (-1, n_mels))
    length = tf.cast(tf.shape(mel_fbanks)[0], tf.int32)
    length = tf.expand_dims(length, 0)
    example['inputs'] = mel_fbanks
    example['inputs_length'] = length
    example.pop('waveforms', None)
    return example


def get_dataset(
    file,
    batch_size=16,
    shuffle_size=16,
    thread_count=24,
    maxlen_feature=1800,
):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'waveforms': tf.float32,
                'targets': tf.int32,
                'targets_length': tf.int32,
            },
            output_shapes={
                'waveforms': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            args=(file,),
        )
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.map(
            preprocess_inputs, num_parallel_calls=thread_count
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'inputs': tf.TensorShape([None, n_mels]),
                'inputs_length': tf.TensorShape([None]),
                'targets': tf.TensorShape([None]),
                'targets_length': tf.TensorShape([None]),
            },
            padding_values={
                'inputs': tf.constant(0, dtype=tf.float32),
                'inputs_length': tf.constant(0, dtype=tf.int32),
                'targets': tf.constant(0, dtype=tf.int32),
                'targets_length': tf.constant(0, dtype=tf.int32),
            },
        )
        return dataset

    return get


def model_fn(features, labels, mode, params):
    conformer_model = conformer.Model(
        kernel_regularizer=None, bias_regularizer=None, **config
    )
    decoder_config = malaya_speech.config.conformer_small_decoder_config
    transducer_model = transducer.rnn.Model(
        conformer_model, vocabulary_size=subwords.vocab_size, **decoder_config
    )
    targets_length = features['targets_length'][:, 0]
    v = tf.expand_dims(features['inputs'], -1)
    z = tf.zeros((tf.shape(features['targets'])[0], 1), dtype=tf.int32)
    c = tf.concat([z, features['targets']], axis=1)

    logits = transducer_model([v, c, targets_length + 1], training=True)

    cost = transducer.loss.rnnt_loss(
        logits=logits,
        labels=features['targets'],
        label_length=targets_length,
        logit_length=features['inputs_length'][:, 0]
        // conformer_model.conv_subsampling.time_reduction_factor,
    )
    mean_error = tf.reduce_mean(cost)

    loss = mean_error

    tf.identity(loss, 'train_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train.optimizer.optimize_loss(
            loss,
            tf.train.AdamOptimizer,
            parameters['optimizer_params'],
            learning_rate_scheduler,
            summaries=['learning_rate', 'loss_scale'],
            larc_params=parameters.get('larc_params', None),
            loss_scaling=parameters.get('loss_scaling', 1.0),
            loss_scaling_params=parameters.get('loss_scaling_params', None),
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [tf.train.LoggingTensorHook(['train_loss'], every_n_iter=1)]
train_dataset = get_dataset('mixed-asr-train.json')

train.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='conformer-rnnt-small-tts',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=25000,
    max_steps=200000,
    train_hooks=train_hooks,
)
