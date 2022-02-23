from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     DepthwiseConv2D, Flatten, PReLU, add)
from tensorflow.keras.regularizers import l2


def conv_block(inputs, filters, kernel_size, strides, padding, weight_decay=5e-4):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, 
               kernel_initializer=initializers.RandomNormal(stddev=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(inputs)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)
    return x

def depthwise_conv_block(inputs, filters, kernel_size, strides, weight_decay=5e-4):
    x = DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False,
                        depthwise_initializer=initializers.RandomNormal(stddev=0.1),
                        depthwise_regularizer=l2(weight_decay), 
                        bias_initializer='zeros')(inputs)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)
    return x

def bottleneck(inputs, filters, kernel, t, strides, r=False, weight_decay=5e-4):
    tchannel = K.int_shape(inputs)[-1] * t
    x = conv_block(inputs, tchannel, 1, 1, "same")

    x = DepthwiseConv2D(kernel, strides=strides, padding="same", depth_multiplier=1, use_bias=False,
                        depthwise_initializer=initializers.RandomNormal(stddev=0.1),
                        depthwise_regularizer=l2(weight_decay), 
                        bias_initializer='zeros')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)
    
    x = Conv2D(filters, 1, strides=1, padding="same", use_bias=False, 
               kernel_initializer=initializers.RandomNormal(stddev=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    if r:
        x = add([x, inputs])
    return x

def inverted_residual_block(inputs, filters, kernel, t, n, weight_decay=5e-4):
    x = inputs
    for _ in range(n):
        x = bottleneck(x, filters, kernel, t, 1, True, weight_decay=weight_decay)
    return x

def mobilefacenet(inputs, embedding_size, weight_decay=5e-4):
    x = conv_block(inputs, 64, 3, 2, "same", weight_decay=weight_decay)  # Output Shape: (56, 56, 64)
    x = depthwise_conv_block(x, 64, 3, 1, weight_decay=weight_decay) # (56, 56, 64)

    x = bottleneck(x, 64, 3, t=2, strides=2, weight_decay=weight_decay)
    x = inverted_residual_block(x, 64, 3, t=2, n=4, weight_decay=weight_decay)  # (28, 28, 64)

    x = bottleneck(x, 128, 3, t=4, strides=2, weight_decay=weight_decay)  # (14, 14, 128)
    x = inverted_residual_block(x, 128, 3, t=2, n=6, weight_decay=weight_decay)  # (14, 14, 128)
    
    x = bottleneck(x, 128, 3, t=4, strides=2, weight_decay=weight_decay)  # (14, 14, 128)
    x = inverted_residual_block(x, 128, 3, t=2, n=2, weight_decay=weight_decay)  # (7, 7, 128)
    
    x = Conv2D(512, 1, use_bias=False,
               kernel_initializer=initializers.RandomNormal(stddev=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = PReLU(alpha_initializer=initializers.constant(0.25), alpha_regularizer=l2(weight_decay), shared_axes=[1, 2])(x)
    
    x = DepthwiseConv2D(int(x.shape[1]), depth_multiplier=1, use_bias=False,
                        depthwise_initializer=initializers.RandomNormal(stddev=0.1),
                        depthwise_regularizer=l2(weight_decay), 
                        bias_initializer='zeros')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    
    x = Conv2D(embedding_size, 1, use_bias=False,
               kernel_initializer=initializers.RandomNormal(stddev=0.1),
               kernel_regularizer=l2(weight_decay),
               bias_initializer='zeros')(x)
    x = BatchNormalization(name="embedding", epsilon=1e-5)(x)
    x = Flatten()(x)

    return x
