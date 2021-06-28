# -*- coding: utf-8 -*-
import tensorflow as tf
import argparse
import os
import glob
import cv2
import config
import numpy as np
from prepare_data import load_and_preprocess_image, generate_datasets
from train import get_model
from config import image_height, image_width, channels
import infer

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import os

dirPath = os.path.dirname(os.path.realpath(__file__))


def read_class_label(label_info_file):
    # 根据文件路径，按行读取类别属性数据

    lines = open(label_info_file).readlines()
    # 去除每行的前后空格
    lines = [line.strip() for line in lines]
    result_dict = {}
    # 对于每一行数据，按空格划分，提取类别ID和属性信息，并按类别ID存放字典类型中返回
    for line in lines:
        if len(line) == 0:
            continue
        # 读取出的信息如“1 0 AM_General_Hummer_SUV_2000”
        # 对每一行按空格划分
        parts_0 = line.split(' ')
        # 提取每一行的训练时的类别ID
        class_id = int(parts_0[1])
        # 提取每一行对应的属性信息
        attribute = parts_0[2]
        if class_id not in result_dict:
            result_dict[class_id] = []
        # 将属性信息按类别ID放入字典中
        result_dict[class_id].append(attribute)
    return result_dict


# 按照路径加载图片
def read_image(img_path):
    # 调用prepare_data.py中的load_and_preprocess_image_for_test函数，加载图像便于进行训练
    img = load_and_preprocess_image(img_path=img_path)
    # 调用np.zeros( )函数，声明一个（n,h,w,c）格式的tensor，数据类型为np.float32，使其能够用于前向推理
    image = np.zeros((1, image_height, image_width, channels), dtype=np.float32)
    # 将img中的数据拷贝到image中。img是与image通道数量相同的图片。
    image[0, :, :, :] = img
    # 加载图像，并resize便于使用同一尺寸进行可视化展示
    image_raw = cv2.imread(img_path)
    # 调用cv2.resize（ ）函数，采用区域插值方法，进行resize便于使用同一尺寸（640, 450）进行可视化展示
    resized_image = cv2.resize(image_raw, (640, 450), interpolation=cv2.INTER_AREA)
    # 返回image 和resized_image
    return image, resized_image


# 在attribute_dict中读取class_id，然后写在img图片上
def write_attribute_to_img(attribute_dict, class_id, img):
    # 根据类别id获取列表属性
    attribute = attribute_dict[class_id]
    # 设置文字相对于图形左上角的偏移位置
    offset = 20
    # 输出识别出的车辆类别标签和对应的属性信息
    print("class_id :", class_id)
    print("attribute:", attribute[0])
    # 调用cv2.putTex函数，将属性信息写入图像中进行可视化，设置的字体如：FONT_HERSHEY_SIMPLEX、FONT_HERSHEY_COMPLEX、FONT_HERSHEY_PLAIN 、FONT_HERSHEY_COMPLEX、FONT_ITALIC、FONT_HERSHEY_DUPLEX 、FONT_HERSHEY_TRIPLEX等
    # 调用cv2.putTex函数，将属性信息写入图像中，添加的文字为attribute:，位置为(offset, offset), 字体为cv2.FONT_HERSHEY_DUPLEX，字体大小/颜色/字体粗细分别为0.60，(255 ,0, 0)、1；
    cv2.putText(img, "attribute:", (offset, offset), cv2.FONT_HERSHEY_DUPLEX, 0.60, color=(255, 0, 0), thickness=1)
    # 调用cv2.putTex函数，将属性信息写入图像中，位置为(offset+100, offset), 字体为cv2.FONT_HERSHEY_TRIPLEX，字体大小/颜色/字体粗细分别为0.50，(255 ,0, 0)、1；
    cv2.putText(img, attribute[0], (offset + 100, offset), cv2.FONT_HERSHEY_TRIPLEX, 0.50, color=(255, 0, 0),
                thickness=1)
    # 返回图像img
    return img,class_id,attribute[0]


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description=" car_attribute ")
    parser.add_argument(
        "--model_path", help="Path to saved model directory",
        default="./saved_checkpoints/model_398.h5")
    parser.add_argument(
        "--visual_img_path", help="Path to visual image path directory",
        default="./visualization/")
    parser.add_argument(
        "--save_visual_path", help="Path to save visual result directory",
        default="./visualization_output")
    return parser.parse_args()


##############################
#     TensorFlow Serving     #
##############################
if __name__ == '__main__':
    tf.app.flags.DEFINE_string('server', 'localhost:9000',
                               'PredictionService host:port')
    FLAGS = tf.app.flags.FLAGS

    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    model_name = 'model_398.h5'  # 也许需要转换成pb 的格式

    # Send request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs['x'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_data[0], shape=[1, 28, 28, 1]))
    future = stub.Predict.future(request, 2.0)
    result = future.result()

    # The predict result
    print(result)
    print(result.outputs)

    # val_list = result.outputs['y'].float_val
    # print('the predict number : {}'.format((np.where(val_list == np.max(val_list)))[0]))


    # 调用read_image函数，读取图像进行预处理
    image, image_raw = read_image(img_path)
    # 将加载的图像输入网络，进行前向推理，得出网络分类输出，预测结果包含了对196个类别的评分
    predictions = model(image, training=False)
    # 从预测结果中，选择最大一个作为类别ID
    class_id = np.argmax(predictions, axis=1)

    # 识别结果的写入图像。根据识别类别ID和读取的属性文件信息，将属性信息写到图像中
    image_visual, id, attribute = write_attribute_to_img(attribute_dict, class_id[0], image_raw)
    # 保存图片
    (filepath, tempfilename) = os.path.split(img_path)
    (filename, extension) = os.path.splitext(tempfilename)
    out_img_name = os.path.join(save_visual_path, filename + '.jpg')
    cv2.imwrite(out_img_name, image_visual)