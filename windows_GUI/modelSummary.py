# -*- coding: utf-8 -*-
import tensorflow as tf
from models.residual_block import build_res_block_2
from config import NUM_CLASSES
from models import resnet50
import config
if __name__ == '__main__':
    
    # 创建ResNet50实例
    model = resnet50.ResNet50(weight_decay = 0.00001, dropout_rate = 0.1)
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    
    #打印网络模型
   

