# 引入库文件
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


# 如果是通过python train.py主动调用该python文件（而不是由其他python文件加载了该文件）
if __name__ == '__main__':

    # -------------teacher_net load----------------- 
    # GPU 设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 判断保存模型的文件夹是否存在
    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)

    # 获取original_dataset训练集
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count,dataset_cat = generate_datasets()

    # 创建模型
    model = get_model()

    # 定义 loss 和 optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # 定义边界
    boundaries = [70, 180, 300]

    # 定义学习率
    values = [0.1, 0.01, 0.001, 0.0001]

    # print('boundaries:', boundaries)
    # print('lrs:', values)

    # 选择优化器，本例使用SGD优化器
    lr_schedules = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.save_model_dir, 'model_{epoch}.h5'), save_best_only=True,
        monitor='val_sparse_categorical_accuracy', verbose=1)

    model.load_weights('saved_checkpoints/model_398.h5')
     

    model.compile(optimizer = optimizer, 
                       loss = 'sparse_categorical_crossentropy',
                    metrics = ['sparse_categorical_accuracy'])
     
    # -------------teacher_net load----------------- 
    teacher_net=model
    student_net = StudentNet()
    
    # -------------student_net ---------------------
        
    batch_size = 32
    epochs = 1

    optimizer = tf.keras.optimizers.Adam()
    train_acc = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
    test_acc = tf.keras.metrics.CategoricalAccuracy('test_accuracy')
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    
    student_net.load_weights('student_net.h5')

    def loss_fn_kd(outputs, labels, teacher_outputs, T=5, alpha=0.1): #5,0.4
        hard_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs) * (1. - alpha)
        KL = tf.keras.losses.KLDivergence()
        soft_loss = KL(tf.nn.softmax(teacher_outputs/T, axis=1), tf.nn.softmax(outputs, axis=1)) * (alpha * T *T)
        print('loss_fn_kd')
        return hard_loss + soft_loss
  
    
    def train_step(images, hard_labels, step, alpha=0.1):
        soft_labels = teacher_net(images,training=False)
        with tf.GradientTape() as tape:   
            logits = student_net(images, training=True)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 5, alpha)
        gradients = tape.gradient(loss, student_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_net.trainable_variables)) 
        train_acc(hard_labels, tf.nn.softmax(logits))
        train_loss(loss)
        print(loss)
        return loss
    
    @tf.function    
    def train():
        for epoch in range(epochs):
            step = 0
            for images, labels in dataset_cat:
                # labels=tf.keras.utils.to_categorical(labels,num_classes=196,dtype='float32')
                loss = train_step(images, labels, step)
                step += 1
                #print(loss)
                tf.print('epoch:',epoch,'step:',step,'train acc:',train_acc.result(),'loss:',loss)
    student_net.load_weights('student_net.h5')
    train()
    student_net.save('student_net.h5')

