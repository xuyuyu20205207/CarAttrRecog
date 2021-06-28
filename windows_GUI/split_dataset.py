import os
import random
import shutil
import config

# 定义分类工具类
class SplitDataset():
    def __init__(self, dataset_dir, saved_dataset_dir, train_ratio=0.8, test_ratio=0.1, show_progress=False):

        # 由成员变量持有文件夹路径
        self.dataset_dir = dataset_dir
        self.saved_dataset_dir = saved_dataset_dir
        #根据传递进来的原始数据的路径参数和保存数据集的路径参数，指定划分数据集train、valid和test的保存地址，训练，验证，测试集的数据分别保存在self.saved_dataset_dir文件夹中
        # ===== 待完善 =====
        self.saved_train_dir = 
        self.saved_valid_dir = 
        self.saved_test_dir = 

        #据传递进来的三个划分数据集图像数量的比例，生成类自身对应的图像数据集的比例（其实这里，对传递进来的参数重新定义可以更方便使用）
        # 记录数据比例
        # 三个数据集valid、train、test的和等于1
        # ===== 待完善 =====
        self.train_ratio = 
        self.test_ratio = 
        self.valid_ratio = 

        # 初始化成员变量
        self.train_file_path = []
        self.valid_file_path = []
        self.test_file_path = []

        self.index_label_dict = {}

        #初始化是否显示处理进度变量
        self.show_progress = show_progress

        # 判断目标文件路径是否存在，如果不存在，则创建文件夹
        if not os.path.exists(self.saved_dataset_dir):
            os.mkdir(self.saved_dataset_dir)
        if not os.path.exists(self.saved_train_dir):
            os.mkdir(self.saved_train_dir)
        if not os.path.exists(self.saved_test_dir):
            os.mkdir(self.saved_test_dir)
        if not os.path.exists(self.saved_valid_dir):
            os.mkdir(self.saved_valid_dir)

    # 获取标签数据的label名称。由于在早先的数据集生成过程中，label是按照名称保存在对应文件夹中，因此这一步相当于是根据文件夹名称进行判断
    #从原始图像数据集中，检索返回所有图像的196类的类别标签
    def __get_label_types(self):
        label_names = []
        # 遍历self.dataset_dir目录
        for item in os.listdir(self.dataset_dir):
            # 获取完整路径
            item_path = os.path.join(self.dataset_dir, item)
            # 判断是不是文件夹
            if os.path.isdir(item_path):
                # 如果是文件夹，则拼接到label_names中
                label_names.append(item)
        # 返回查找结果
        return label_names

    # 获取所有文件名，获得所有图形文件的路径
    def __get_all_file_path(self):
        all_file_path = []
        index = 0
        # 获取所有的标签类别
        for file_type in self.__get_label_types():
            # 由index_label_dict数组持有label
            self.index_label_dict[index] = file_type
            index += 1
            # 拼接全路径
            type_file_path = os.path.join(self.dataset_dir, file_type)
            file_path = []
            # 遍历目标路径文件夹
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(type_file_path, file)
                # 将路径文件夹下的文件加入数组file_path
                file_path.append(single_file_path)
            # 将本次遍历的目标路径file_path加入数组all_file_path，最终all_file_path是一个二维数组
            all_file_path.append(file_path)
        return all_file_path

    # 将type_path中的文件以此拷贝到type_saved_dir中
    def __copy_files(self, type_path, type_saved_dir):
        for item in type_path:
            src_path_list = item[1]
            dst_path = type_saved_dir + "%s/" % (item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            for src_path in src_path_list:
                # ===== 待完善 =====
                shutil.copy( , )
                if self.show_progress:
                    print("Copying file "+src_path+" to "+dst_path)

    # 将训练数据拆分成train、test、valid
    def __split_dataset(self):
        # 获取所有训练图片，存在二维数组中，第一维是所有子路径（以label type命名），第二维是每个子路径对应的所有训练图片
        # ===== 待完善 =====
        all_file_paths = 
        # 遍历每个子路径
        for index in range(len(all_file_paths)):
            file_path_list = all_file_paths[index]
            file_path_list_length = len(file_path_list)
            # 将该子路径下的文件随机乱序以提高训练效果
            random.shuffle(file_path_list)

            # 计算训练集和测试集中应有的图片数目
            # ===== 待完善 =====
            train_num = int(  )
            test_num = int(  )

            # 将图片按指定数目存放到目标路径数组
            self.train_file_path.append([self.index_label_dict[index], file_path_list[: train_num]])
            self.test_file_path.append([self.index_label_dict[index], file_path_list[train_num:train_num + test_num]])
            self.valid_file_path.append([self.index_label_dict[index], file_path_list[train_num + test_num:]])

    #定义数据集划分函数，应用之前定义的类函数，实现划分功能
    def start_splitting(self):
        # 拆分数据，调用划分数据集函数，获得三个数据集的路径
        # ===== 待完善 =====
        self.__split_dataset()
        # 拷贝文件
        #拷贝完成训练集、验证集、测试集的划分
        # ===== 待完善 =====
        self.__copy_files(type_path=, type_saved_dir=)
        self.__copy_files(type_path=, type_saved_dir=)
        self.__copy_files(type_path=, type_saved_dir=r)

# 如果是主动调用该脚本（而不是由其他的脚本呼起该脚本），则执行下面的语句
if __name__ == '__main__':
    # #指定原始图像数据集的路径和三个数据集划分后的保存路径，声明划分类，初始化时候，原始数据集路径为config.dataset_origin_dir，保存路径设置为config.dataset_dir
    # ===== 待完善 =====
    split_dataset = SplitDataset(dataset_dir=,
                                 saved_dataset_dir=,
                                 show_progress=True)
    # 执行分类过程
    split_dataset.start_splitting()
