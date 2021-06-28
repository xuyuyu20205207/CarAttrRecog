from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models import resnet50, resnet101, resnet152, resnet34, resnet18
import config
from prepare_data import generate_datasets
import math
import os
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
from StudentNet import StudentNet
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile

def get_model():
    # 默认resnet50
    model = resnet50.ResNet50(weight_decay=0.00001, dropout_rate=0.1)
    if config.model == "resnet18":
        model = resnet18.ResNet18(weight_decay=0.00001, dropout_rate=0.1)
    if config.model == "resnet34":
        model = resnet34.ResNet34(weight_decay=0.00001, dropout_rate=0.1)
    if config.model == "resnet101":
        model = resnet101.ResNet101()
    if config.model == "resnet152":
        model = resnet152.ResNet152()

    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    return model

if __name__=='__main__':    
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count,dataset_cat = generate_datasets()


    teacher_net=get_model()
    teacher_net.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['sparse_categorical_accuracy'])
    teacher_net.load_weights('saved_checkpoints/model_398.h5')
    # _, teacher_net_accuracy = teacher_net.evaluate(test_dataset, verbose=0)
    # print('teacher net acc:',teacher_net_accuracy)
    #
    #
    # student_net=StudentNet()
    # student_net.load_weights('student_net.h5')
    # student_net.compile(optimizer='adam',
    #                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                           metrics=['sparse_categorical_accuracy'])
    # _, student_net_accuracy = student_net.evaluate(test_dataset, verbose=0)
    # print('student net acc:',student_net_accuracy)
    #
    #
    # model_for_export = tfmot.sparsity.keras.strip_pruning(student_net)
    # model_for_export.summary()
    #
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    # pruned_tflite_model = converter.convert()
    #
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    # # Auto Weight Quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # quantized_and_pruned_tflite_model = converter.convert()
    # with open('model.tflite', 'wb') as f:
    #     f.write(quantized_and_pruned_tflite_model)
    #
    #
    # print('ok')
