import os

import matplotlib.pyplot as plt
import numpy as np

from nets.arcface import arcface
from utils.utils import preprocess_input, resize_image, show_config


class Arcface(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        #--------------------------------------------------------------------------#
        "model_path"        : "model_data/arcface_mobilefacenet.h5",
        #-------------------------------------------#
        #   输入图片的大小。
        #-------------------------------------------#
        "input_shape"       : [112, 112, 3],
        #-------------------------------------------#
        #   所使用到的主干特征提取网络，与训练的相同
        #   mobilefacenet
        #   mobilenetv1
        #   iresnet50
        #-------------------------------------------#
        "backbone"          : "mobilefacenet",
        #-------------------------------------------#
        #   是否进行不失真的resize
        #-------------------------------------------#
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Arcface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        
        show_config(**self._defaults)
        
    def generate(self):
        #---------------------------------------------------#
        #   载入模型与权值
        #---------------------------------------------------#
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model = arcface(self.input_shape, backbone=self.backbone, mode="predict")

        print('Loading weights into state dict...')
        self.model.load_weights(self.model_path, by_name=True)
        print('{} model loaded.'.format(self.model_path))
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        
        photo_1 = np.expand_dims(preprocess_input(np.array(image_1, np.float32)), 0)
        photo_2 = np.expand_dims(preprocess_input(np.array(image_2, np.float32)), 0)

        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        output1 = self.model.predict(photo_1)
        output2 = self.model.predict(photo_2)
    
        #---------------------------------------------------#
        #   计算二者之间的距离
        #---------------------------------------------------#
        l1 = np.linalg.norm(output1-output2, axis=1)
        # l1 = np.sum(np.square(output1 - output2), axis=-1)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom',fontsize=11)
        plt.show()
        return l1

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = resize_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        
        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        preds       = self.model.predict(image_data)[0]
        import time
        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds       = self.model.predict(image_data)[0]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time