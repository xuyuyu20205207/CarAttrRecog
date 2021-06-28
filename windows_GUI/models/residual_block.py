
# 定义基础残差网络

import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, weight_decay = 0.001):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            kernel_initializer = 'he_normal',
                                            kernel_regularizer = tf.keras.regularizers.l2(weight_decay))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            kernel_initializer = 'he_normal',
                                            kernel_regularizer = tf.keras.regularizers.l2(weight_decay))
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride,
                                                       kernel_initializer = 'he_normal',
                                                       kernel_regularizer = tf.keras.regularizers.l2(weight_decay)))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu = tf.nn.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2, training=training)

        output = tf.nn.relu(tf.keras.layers.add([identity, bn2]))

        return output

# 残差网络卷积模块
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1, weight_decay = 0.01):
        super(BottleNeck, self).__init__()
        # 依次定义CONV2D层、BatchNorm层
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer = 'he_normal',
                                            kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
                                            )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer = 'he_normal',
                                            kernel_regularizer = tf.keras.regularizers.l2(weight_decay))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer = 'he_normal',
                                            kernel_regularizer = tf.keras.regularizers.l2(weight_decay))
        self.bn3 = tf.keras.layers.BatchNormalization()

        # 定义Identity Block
        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride,
                                                   kernel_initializer = 'he_normal',
                                                   kernel_regularizer = tf.keras.regularizers.l2(weight_decay)))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        # 使用初始化的组件，搭建了卷积模块
        identity = self.downsample(inputs)

        # 第一层conv+batchnorm+relu
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu1 = tf.nn.relu(bn1)

        # 第二层conv+batchnorm+relu
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2, training=training)
        relu2 = tf.nn.relu(bn2)

        # 第三层conv+ relu
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3, training=training)

        # 将Identity Block和Conv Block连接作为输出
        output = tf.nn.relu(tf.keras.layers.add([identity, bn3]))

        return output

def build_res_block_1(filter_num, blocks, stride=1, weight_decay = 0.001):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride, weight_decay = weight_decay))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1, weight_decay = weight_decay))

    return res_block


def build_res_block_2(filter_num, blocks, stride=1, weight_decay = 0.01):
    res_block = tf.keras.Sequential()
    # 创建第一个BottleNeck时指定stride，其余使用1
    res_block.add(BottleNeck(filter_num, stride=stride, weight_decay = weight_decay))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1, weight_decay = weight_decay))

    return res_block
