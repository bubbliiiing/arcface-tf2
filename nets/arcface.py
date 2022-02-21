import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Input, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from nets.iresnet import iResNet50
from nets.mobilefacenet import mobilefacenet
from nets.mobilenet import MobilenetV1


class ArcMarginProduct(Layer) :
    def __init__(self, n_classes=1000, **kwargs) :
        self.n_classes = n_classes
        super(ArcMarginProduct, self).__init__(**kwargs)

    def build(self, input_shape) :
        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.n_classes),
                                initializer=initializers.glorot_uniform(),
                                trainable=True,
                                regularizer=l2(5e-4))
        super(ArcMarginProduct, self).build(input_shape)
        
    def call(self, input) :
        W       = tf.nn.l2_normalize(self.W, axis=0)
        logits  = input @ W
        return K.clip(logits, -1 + K.epsilon(), 1 - K.epsilon())

    def compute_output_shape(self, input_shape) :
        return (None, self.n_classes)

def arcface(input_shape, num_classes=None, backbone="mobilefacenet", mode="train", weight_decay=5e-4):
    inputs = Input(shape=input_shape)

    if backbone=="mobilefacenet":
        embedding_size  = 128
        x = mobilefacenet(inputs, embedding_size, weight_decay=weight_decay)
    elif backbone=="mobilenetv1":
        embedding_size  = 512
        x = MobilenetV1(inputs, embedding_size, dropout_keep_prob=0.5, weight_decay=weight_decay)
    elif backbone=="iresnet50":
        embedding_size  = 512
        x = iResNet50(inputs, embedding_size, dropout_keep_prob=0.5, weight_decay=weight_decay)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1, iresnet50.'.format(mode))

    if mode == "train":
        predict = Lambda(lambda  x: K.l2_normalize(x, axis=1), name="l2_normalize")(x)
        x       = ArcMarginProduct(num_classes, name="ArcMargin")(predict)
        model   = Model(inputs, [x, predict])
        return model
    elif mode == "predict":
        x       = Lambda(lambda  x: K.l2_normalize(x, axis=1))(x)
        model   = Model(inputs, x)
        return model
    else:
        raise ValueError('Unsupported mode - `{}`, Use train, predict.'.format(mode))
