# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models import resnet50, resnet101, resnet152, resnet34
import config
import numpy as np

if __name__ == '__main__':
    #新建resnet网络模型实例
    model = resnet50.ResNet50()
    model.build(input_shape = (None, config.image_height, config.image_width, 3))
    
    # 调用numpy中的方法，升恒随机数据，模拟符合条件的网络输入图像
    random_data = 
    # 将模拟数据转换为符合网络输入要求的tensor
    tf_data = 
    # 将模拟数据输入网络，获得网络处理结果
    preds =  model( )
    # 打印模拟数据的网络预测结果 
    print(preds)

