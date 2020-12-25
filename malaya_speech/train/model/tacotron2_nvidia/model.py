import tensorflow as tf
from .encoder import Tacotron2Encoder
from .decoder import Tacotron2Decoder

encoder_config = {
    'cnn_dropout_prob': 0.5,
    'rnn_dropout_prob': 0.0,
    'src_emb_size': 512,
    'conv_layers': [
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
        },
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
        },
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
        },
    ],
    'activation_fn': tf.nn.relu,
    'num_rnn_layers': 1,
    'rnn_cell_dim': 256,
    'rnn_unidirectional': False,
    'use_cudnn_rnn': False,
    'rnn_type': tf.nn.rnn_cell.LSTMCell,
    'zoneout_prob': 0.0,
    'data_format': 'channels_last',
    'dtype': tf.float32,
    'regularizer': tf.contrib.layers.l2_regularizer,
    'regularizer_params': {'scale': 1e-6},
    'initializer': tf.contrib.layers.xavier_initializer,
}

decoder_config = {
    'zoneout_prob': 0.0,
    'dropout_prob': 0.1,
    'attention_type': 'location',
    'attention_layer_size': 128,
    'attention_bias': True,
    'decoder_cell_units': 1024,
    'decoder_cell_type': tf.nn.rnn_cell.LSTMCell,
    'decoder_layers': 2,
    'enable_prenet': True,
    'prenet_layers': 2,
    'prenet_units': 256,
    'enable_postnet': True,
    'postnet_keep_dropout_prob': 0.7,
    'postnet_data_format': 'channels_last',
    'postnet_conv_layers': [
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'activation_fn': tf.nn.tanh,
        },
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'activation_fn': tf.nn.tanh,
        },
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'activation_fn': tf.nn.tanh,
        },
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': 512,
            'padding': 'SAME',
            'activation_fn': tf.nn.tanh,
        },
        {
            'kernel_size': [5],
            'stride': [1],
            'num_channels': -1,
            'padding': 'SAME',
            'activation_fn': None,
        },
    ],
    'mask_decoder_sequence': True,
    'parallel_iterations': 32,
    'dtype': tf.float32,
    'regularizer': tf.contrib.layers.l2_regularizer,
    'regularizer_params': {'scale': 1e-6},
    'initializer': tf.contrib.layers.xavier_initializer,
}


class Model:
    def __init__(
        self,
        encoder_inputs,
        decoder_inputs,
        vocab_size,
        training = True,
        **kwargs
    ):
        if training:
            mode = 'train'
        else:
            mode = 'eval'

        e_config = {**encoder_config, **kwargs}
        d_config = {**decoder_config, **kwargs}

        e_config['src_vocab_size'] = vocab_size
        d_config['num_audio_features'] = int(decoder_inputs[0].shape[-1])

        self.encoder = Tacotron2Encoder(e_config, None, mode = mode)
        input_dict = {'source_tensors': encoder_inputs}
        self.encoder_logits = self.encoder.encode(input_dict)

        input_dict['encoder_output'] = self.encoder_logits
        input_dict['target_tensors'] = decoder_inputs

        self.decoder = Tacotron2Decoder(d_config, None, mode = mode)
        self.decoder_logits = self.decoder.decode(input_dict)
