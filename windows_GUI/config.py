# -*- coding: utf-8 -*-

# 学习回合数，表示整个训练过程要遍历多少次训练集
EPOCHS = 1

# 每次计算损失loss使用的训练图片数量。使用小的batchsize能够节省内存，但是过小的batchsize可能会导致训练参数的震荡，过大的batchsize容易让模型陷入局部最优解，降低模型的泛化性能。本课程中使用的batchsize默认设置为32，即每一次训练会同时给定32张图片。训练将会计算这32张图下的loss之和，然后更新模型参数。
# 提高该数值，会增加对GPU内存的占用
BATCH_SIZE = 32

# 分类数目
NUM_CLASSES = 196

# 训练图片的宽和高
image_height = 224
image_width = 224

# 训练和预测时会将读取到的图片resize到宽高等于new_size。在tensorflow执行random_crop的时候，会从new_size*new_size的图片中裁剪出image_height*image_width大小的图片用于训练
new_size = 256

# 图片的通道数目，RGB图片为3通道，灰度图为1通道
channels = 3

# 保存model的文件夹名称
save_model_dir = "./saved_checkpoints"

# 数据来源目录，存放的是未经split_dataset.py处理的数据
dataset_origin_dir = "./standford_cars/origin_dataset"

# 数据集来源目录。由split_dataset.py生成
dataset_dir = "dataset"

# 训练集目录
train_dir = dataset_dir + "/train"
# 交叉验证集目录
valid_dir = dataset_dir + "/valid"
# 测试集目录
test_dir = dataset_dir + "/test"


# 训练集的预处理，进行归一化，以便使网络精度更高。归一化就是-均值/方差
mean = [0.485, 0.456, 0.406] #均值
std = [0.229, 0.224, 0.225] #方差

# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
