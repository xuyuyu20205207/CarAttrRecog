import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Activation, GlobalAveragePooling2D
        
def StudentNet():
    multiplier = [1, 2, 4, 8, 16, 16, 16, 16]
    bandwidth = [ 16 * m for m in multiplier]
    StudentNet = Sequential([
       	# Standard Convolution
        Conv2D(
        filters=bandwidth[0], 
        kernel_size=(3, 3),
                input_shape=(224, 224, 3), padding='same'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        MaxPool2D(pool_size=(2, 2), strides=2), 
# Depthwise Separable Convolution                          
        Conv2D(bandwidth[0], 3, padding='same'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        Conv2D(bandwidth[1], 1, padding='same'),
        MaxPool2D(2, 2),
        # Depthwise Separable Convolution
        Conv2D(bandwidth[1], 3, padding='same'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        Conv2D(bandwidth[2], 1),
        MaxPool2D(2, 2),
        
        Conv2D(bandwidth[2], 3, padding='same'),
        BatchNormalization(),
        Activation(tf.nn.relu6),
        Conv2D(bandwidth[3], 1, padding='same'),
        MaxPool2D(2, 2),
        
        Conv2D(bandwidth[3], 3, padding='same'),
        Activation(tf.nn.relu),
        Conv2D(bandwidth[4], 1, padding='same'),
        
        Conv2D(bandwidth[4], 3,  padding='same'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        Conv2D(bandwidth[5], 1),
        
        Conv2D(bandwidth[5], 3,  padding='same'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        Conv2D(bandwidth[6], 1, padding='same'),
       
        Conv2D(bandwidth[6], 3, padding='same'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        Conv2D(bandwidth[7], 1, padding='same'),

        GlobalAveragePooling2D(),
        Dense(196)
    ])
    return StudentNet
