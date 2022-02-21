
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, DepthwiseConv2D, Dropout, Flatten,
                                     PReLU)
from tensorflow.keras.regularizers import l2


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1), weight_decay=5e-4):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1',
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(inputs)
    x = BatchNormalization(name='conv1_bn', epsilon=1e-5)(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1, weight_decay=5e-4):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id,
                        depthwise_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                        depthwise_regularizer=l2(weight_decay), 
                        bias_initializer='zeros')(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id, epsilon=1e-5)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id,
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id, epsilon=1e-5)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    return K.relu(x, max_value=6)

def MobilenetV1(inputs, embedding_size, dropout_keep_prob=0.5, depth_multiplier=1, weight_decay=5e-4):
    x = _conv_block(inputs, 32, strides=(1, 1))
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1, weight_decay=weight_decay)

    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3, weight_decay=weight_decay)

    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5, weight_decay=weight_decay)

    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11, weight_decay=weight_decay)

    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12, weight_decay=weight_decay)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13, weight_decay=weight_decay)

    x = Conv2D(512, kernel_size=1, use_bias=False, name='sep',
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name='sep_bn', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)

    x = BatchNormalization(name='bn2', epsilon=1e-5)(x)
    x = Dropout(dropout_keep_prob)(x)
    x = Flatten()(x)
    x = Dense(embedding_size, name='linear',
            kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
            kernel_regularizer=l2(weight_decay),
            bias_initializer='zeros')(x)
    x = BatchNormalization(name='features', epsilon=1e-5)(x)
    return x
