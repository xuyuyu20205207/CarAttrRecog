# -*- coding: utf-8 -*-
import tensorflow as tf
import config
import pathlib
import numpy as np
from datetime import datetime

from config import image_height, image_width, channels, new_size,mean, std

def load_and_preprocess_image(img_path):
    # 读取图片
    img_raw = tf.io.read_file(img_path)
    # 解码图片
    img_tensor = tf.image.decode_jpeg(img_raw, channels=channels)
    # 将图片尺寸resize到config中配置的宽高数据
    img_tensor = tf.image.resize(img_tensor, [new_size, new_size])
    # 对图片进行随机切割，是提升算法鲁棒性和数据增强的方式
    img_tensor = tf.image.random_crop(img_tensor, [image_height, image_width, channels])
    img_tensor = tf.image.random_flip_left_right(img_tensor)
    # tf.image.decode_png 得到的是uint8格式，范围在0-255之间，经过convert_image_dtype 就会被转换为区间在0-1之间的float32格式
    img_tensor = tf.image.convert_image_dtype(img_tensor, dtype = tf.float32)
    print('img tensor shape:', img_tensor)
    # 使用均值和方差处理图片
    color_mean = tf.constant(mean, dtype = tf.float32, shape = (1,1,3))
    color_std = tf.constant(std, dtype = tf.float32, shape = (1,1,3))
    img_tensor -= color_mean
    img_tensor /= color_std
    # img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    # img = img_tensor / 255.0
    return img_tensor


def load_and_preprocess_image_for_test(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels = channels)
    img_tensor = tf.image.resize(img_tensor, [new_size, new_size])
    img_tensor = tf.image.resize(img_tensor, [image_height, image_width])

    # tf.image.decode_png 得到的是uint8格式，范围在0-255之间，经过convert_image_dtype 就会被转换为区间在0-1之间的float32格式
    img_tensor = tf.image.convert_image_dtype(img_tensor, dtype = tf.float32)
    # 使用均值和方差处理图片
    color_mean = tf.constant(mean, dtype = tf.float32, shape = (1,1,3))
    color_std = tf.constant(std, dtype = tf.float32, shape = (1,1,3))
    img_tensor -= color_mean
    img_tensor /= color_std
    
    return img_tensor


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((index, label) for label, index in enumerate(label_names))
    print(label_to_index)
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]
    
    return all_image_path, all_image_label


def get_dataset(dataset_root_dir, training = None):
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images
    if training:
        image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
        test_labels = tf.keras.utils.to_categorical(np.array(all_image_label).astype('float32'))
    else:
        image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image_for_test)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)
    
    if training:
        label_dataset_cat=tf.data.Dataset.from_tensor_slices(test_labels)
        dataset_cat = tf.data.Dataset.zip((image_dataset, label_dataset_cat))
        return dataset,image_count,dataset_cat
    
    return dataset, image_count


def generate_datasets(training=True):
    tic = datetime.now()
    print(tic)
    
    train_dataset, train_count,dataset_cat = get_dataset(dataset_root_dir=config.train_dir, training = True)
    valid_dataset, valid_count = get_dataset(dataset_root_dir=config.valid_dir)
    test_dataset, test_count = get_dataset(dataset_root_dir=config.test_dir)

    # 以batch形式读取original_dataset
    # 训练集要打乱顺序
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE).repeat()
    dataset_cat=dataset_cat.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)
    toc = datetime.now()
    print(toc)
    
    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count,dataset_cat
