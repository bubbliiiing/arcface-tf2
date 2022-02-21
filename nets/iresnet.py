from tensorflow.keras import initializers, layers
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, PReLU, ZeroPadding2D)
from tensorflow.keras.regularizers import l2


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=5e-4):
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2', epsilon=1e-5)(input_tensor)

    #----------------------------#
    #   减少通道数
    #----------------------------#
    x = Conv2D(filters1, kernel_size, padding='same', use_bias=False, name=conv_name_base + '2a',
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2a', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)

    #----------------------------#
    #   3x3卷积
    #----------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=conv_name_base + '2b',
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2b', epsilon=1e-5)(x)
    
    x = layers.add([x, input_tensor])
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), weight_decay=5e-4):
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2', epsilon=1e-5)(input_tensor)
    
    #----------------------------#
    #   减少通道数
    #----------------------------#
    x = Conv2D(filters1, kernel_size, padding='same', use_bias=False, name=conv_name_base + '2a',
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2a', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)

    #----------------------------#
    #   3x3卷积
    #----------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, strides=strides, name=conv_name_base + '2b',
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name=bn_name_base + '2b', epsilon=1e-5)(x)
    
    #----------------------------#
    #   残差边
    #----------------------------#
    shortcut = Conv2D(filters2, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '1',
               kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1', epsilon=1e-5)(shortcut)

    x = layers.add([x, shortcut])
    return x

def iResNet50(inputs, embedding_size, dropout_keep_prob=0.5, weight_decay=5e-4):
    x = ZeroPadding2D((1, 1))(inputs)
    x = Conv2D(64, (3, 3), strides=(1, 1), name='conv1', use_bias=False,
                kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                kernel_regularizer=l2(weight_decay),
                bias_initializer='zeros')(x)
    x = BatchNormalization(name='bn_conv1', epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)

    x = conv_block(x, 3, [64, 64], stage=2, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [64, 64], stage=2, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [64, 64], stage=2, block='c', weight_decay=weight_decay)

    x = conv_block(x, 3, [128, 128], stage=3, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128], stage=3, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128], stage=3, block='c', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128], stage=3, block='d', weight_decay=weight_decay)

    x = conv_block(x, 3, [256, 256], stage=4, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='c', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='d', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='e', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='f', weight_decay=weight_decay)

    x = identity_block(x, 3, [256, 256], stage=4, block='g', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='h', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='i', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='j', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='k', weight_decay=weight_decay)

    x = identity_block(x, 3, [256, 256], stage=4, block='l', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='m', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256], stage=4, block='n', weight_decay=weight_decay)

    x = conv_block(x, 3, [512, 512], stage=5, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [512, 512], stage=5, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [512, 512], stage=5, block='c', weight_decay=weight_decay)
    
    x = BatchNormalization(name='bn2', epsilon=1e-5)(x)
    x = Dropout(dropout_keep_prob)(x)
    x = Flatten()(x)
    x = Dense(embedding_size, name='linear',
            kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
            kernel_regularizer=l2(weight_decay),
            bias_initializer='zeros')(x)
    x = BatchNormalization(name='features', epsilon=1e-5,)(x)

    return x
