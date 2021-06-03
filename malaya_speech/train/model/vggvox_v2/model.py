from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Conv1D, Conv2D, Input, Lambda
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K

weight_decay = 1e-4


def identity_block_2D(
    input_tensor, kernel_size, filters, stage, block, trainable=True
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(
        filters1,
        (1, 1),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_1,
    )(input_tensor)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_1
    )(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_2,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_2
    )(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(
        filters3,
        (1, 1),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_3,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_3
    )(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_2D(
    input_tensor,
    kernel_size,
    filters,
    stage,
    block,
    strides=(2, 2),
    trainable=True,
):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(
        filters1,
        (1, 1),
        strides=strides,
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_1,
    )(input_tensor)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_1
    )(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_2,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_2
    )(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(
        filters3,
        (1, 1),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_3,
    )(x)
    x = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_3
    )(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(
        filters3,
        (1, 1),
        strides=strides,
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=trainable,
        kernel_regularizer=l2(weight_decay),
        name=conv_name_4,
    )(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, trainable=trainable, name=bn_name_4
    )(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_2D_v1(inputs, mode='train'):
    bn_axis = 3
    #     if mode == 'train':
    #         inputs = Input(shape=input_dim, name='input')
    #     else:
    #         inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(
        64,
        (7, 7),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=True,
        kernel_regularizer=l2(weight_decay),
        padding='same',
        name='conv1_1/3x3_s1',
    )(inputs)

    x1 = BatchNormalization(
        axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True
    )(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(
        x1,
        3,
        [48, 48, 96],
        stage=2,
        block='a',
        strides=(1, 1),
        trainable=True,
    )
    x2 = identity_block_2D(
        x2, 3, [48, 48, 96], stage=2, block='b', trainable=True
    )

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(
        x2, 3, [96, 96, 128], stage=3, block='a', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [96, 96, 128], stage=3, block='b', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [96, 96, 128], stage=3, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(
        x3, 3, [128, 128, 256], stage=4, block='a', trainable=True
    )
    x4 = identity_block_2D(
        x4, 3, [128, 128, 256], stage=4, block='b', trainable=True
    )
    x4 = identity_block_2D(
        x4, 3, [128, 128, 256], stage=4, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(
        x4, 3, [256, 256, 512], stage=5, block='a', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [256, 256, 512], stage=5, block='b', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [256, 256, 512], stage=5, block='c', trainable=True
    )
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y


def resnet_2D_v2(inputs, mode='train'):
    bn_axis = 3
    #     if mode == 'train':
    #         inputs = Input(shape=input_dim, name='input')
    #     else:
    #         inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        kernel_initializer='orthogonal',
        use_bias=False,
        trainable=True,
        kernel_regularizer=l2(weight_decay),
        padding='same',
        name='conv1_1/3x3_s1',
    )(inputs)

    x1 = BatchNormalization(
        axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True
    )(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(
        x1,
        3,
        [64, 64, 256],
        stage=2,
        block='a',
        strides=(1, 1),
        trainable=True,
    )
    x2 = identity_block_2D(
        x2, 3, [64, 64, 256], stage=2, block='b', trainable=True
    )
    x2 = identity_block_2D(
        x2, 3, [64, 64, 256], stage=2, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(
        x2, 3, [128, 128, 512], stage=3, block='a', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [128, 128, 512], stage=3, block='b', trainable=True
    )
    x3 = identity_block_2D(
        x3, 3, [128, 128, 512], stage=3, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(
        x3,
        3,
        [256, 256, 1024],
        stage=4,
        block='a',
        strides=(1, 1),
        trainable=True,
    )
    x4 = identity_block_2D(
        x4, 3, [256, 256, 1024], stage=4, block='b', trainable=True
    )
    x4 = identity_block_2D(
        x4, 3, [256, 256, 1024], stage=4, block='c', trainable=True
    )
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(
        x4, 3, [512, 512, 2048], stage=5, block='a', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [512, 512, 2048], stage=5, block='b', trainable=True
    )
    x5 = identity_block_2D(
        x5, 3, [512, 512, 2048], stage=5, block='c', trainable=True
    )
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y


class VladPooling(keras.layers.Layer):
    """
    This layer follows the NetVlad, GhostVlad
    """

    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(
            shape=[self.k_centers + self.g_centers, input_shape[0][-1]],
            name='centers',
            initializer='orthogonal',
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers * input_shape[0][-1])

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(
            exp_cluster_score, axis=-1, keepdims=True
        )

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)  # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(
            feat, -2
        )  # feat_broadcast : bz x W x H x 1 x D
        feat_res = (
            feat_broadcast - self.cluster
        )  # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(
            A, feat_res
        )  # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, : self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(
            cluster_l2, [-1, int(self.k_centers) * int(num_features)]
        )
        return outputs


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def vggvox_resnet2d_icassp(
    inputs, num_class=8631, mode='train', args=None
):

    net = 'resnet34s'
    loss = 'softmax'
    vlad_clusters = 8
    ghost_clusters = 2
    bottleneck_dim = 512
    aggregation = 'gvlad'
    mgpu = 0

    if net == 'resnet34s':
        inputs, x = resnet_2D_v1(inputs, mode=mode)
    else:
        inputs, x = resnet_2D_v2(inputs, mode=mode)

    x_fc = keras.layers.Conv2D(
        bottleneck_dim,
        (7, 1),
        strides=(1, 1),
        activation='relu',
        kernel_initializer='orthogonal',
        use_bias=True,
        trainable=True,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay),
        name='x_fc',
    )(x)

    # ===============================================
    #            Feature Aggregation
    # ===============================================
    if aggregation == 'avg':
        if mode == 'train':
            x = keras.layers.AveragePooling2D(
                (1, 5), strides=(1, 1), name='avg_pool'
            )(x)
            x = keras.layers.Reshape((-1, bottleneck_dim))(x)
        else:
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = keras.layers.Reshape((1, bottleneck_dim))(x)

    elif aggregation == 'vlad':
        x_k_center = keras.layers.Conv2D(
            vlad_clusters,
            (7, 1),
            strides=(1, 1),
            kernel_initializer='orthogonal',
            use_bias=True,
            trainable=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='vlad_center_assignment',
        )(x)
        x = VladPooling(
            k_centers=vlad_clusters, mode='vlad', name='vlad_pool'
        )([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = keras.layers.Conv2D(
            vlad_clusters + ghost_clusters,
            (7, 1),
            strides=(1, 1),
            kernel_initializer='orthogonal',
            use_bias=True,
            trainable=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='gvlad_center_assignment',
        )(x)
        x = VladPooling(
            k_centers=vlad_clusters,
            g_centers=ghost_clusters,
            mode='gvlad',
            name='gvlad_pool',
        )([x_fc, x_k_center])

    else:
        raise IOError('==> unknown aggregation mode')
    x = keras.layers.Dense(
        bottleneck_dim,
        activation='relu',
        kernel_initializer='orthogonal',
        use_bias=True,
        trainable=True,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay),
        name='fc6',
    )(x)
    if loss == 'softmax':
        y = keras.layers.Dense(
            num_class,
            activation=None,
            kernel_initializer='orthogonal',
            use_bias=False,
            trainable=True,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='prediction',
        )(x)
        trnloss = 'categorical_crossentropy'

    elif loss == 'amsoftmax':
        x_l2 = keras.layers.Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = keras.layers.Dense(
            num_class,
            kernel_initializer='orthogonal',
            use_bias=False,
            trainable=True,
            kernel_constraint=keras.constraints.unit_norm(),
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name='prediction',
        )(x_l2)
        trnloss = amsoftmax_loss

    else:
        raise IOError('==> unknown loss.')

    return y


class Model:
    def __init__(self, inputs, num_class, mode='train'):

        self.logits = vggvox_resnet2d_icassp(
            inputs, num_class=num_class, mode=mode
        )
