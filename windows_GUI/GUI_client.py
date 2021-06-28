###客户端client.py
import socket
import os
import sys
import struct

import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class filedialogdemo(QWidget):

    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        layout = QHBoxLayout()

        layout_left = QVBoxLayout()
        self.btn = QPushButton()
        self.btn.clicked.connect(self.loadFile)  # 重选
        self.btn.setText("从文件中获取照片")
        layout_left.addWidget(self.btn)
        # layout_left.addStretch(1)

        self.label = QLabel()
        self.label.setText('原\n图\n片\n')
        self.label.setAlignment(Qt.AlignCenter)
        layout_left.addWidget(self.label)
        # layout_left.addStretch(3)
        layout.addLayout(layout_left)
        # layout.addStretch(1)

        layout_right = QVBoxLayout()
        self.btn_1 = QPushButton()
        self.btn_1.clicked.connect(self.saveFile)  # 重选
        self.btn_1.setText("存储图片")
        layout_right.addWidget(self.btn_1)
        # layout_right.addStretch(1)

        self.label_1 = QLabel()
        self.label_1.setText('识\n别\n结\n果')
        self.label_1.setAlignment(Qt.AlignCenter)
        layout_right.addWidget(self.label_1)
        # layout_right.addStretch(3)
        layout.addLayout(layout_right)
        # layout.addStretch(1)

        self.setLayout(layout)

    def loadFile(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image files(*.jpg *.gif *.png)')

        parent_path = os.path.dirname(fname)
        # print('parent_path = %s' % parent_path)
        file_name = os.path.split(fname)[-1]
        # print('file_name = %s' % file_name)
        new_file = 'new_' + file_name

        image_raw = cv2.imread(fname)
        max_len=max(image_raw.shape[0],image_raw.shape[1])
        print(image_raw.shape)
        print(int(image_raw.shape[0]*1.0/max_len*600),int( image_raw.shape[1]*1.0/max_len*600))
        resized_image = cv2.resize(image_raw, (600,480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(new_file, resized_image)

        pixmax = QPixmap(new_file)
        # pixmax.scaled(600, 600)
        self.label.setPixmap(pixmax)
        # self.label.setScaledContents(True)
        pixmax.save(file_name)


        # 检测
        # get_result(file_name)

        # 与服务器通信
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # s.connect(('172.16.0.56', 6666))  # 服务器和客户端在不同的系统或不同的主机下时使用的ip和端口，首先要查看服务器所在的系统网卡的ip
            s.connect(('127.0.0.1', 6666))  #服务器和客户端都在一个系统下时使用的ip和端口
        except socket.error as msg:
            print(msg)
            print(sys.exit(1))
        # filepath = input('input the file: ')  # 输入当前目录下的图片名 xxx.jpg
        filepath=file_name
        # 拿图片
        # 先用input
        # 发图片
        fhead = struct.pack(b'128sq', bytes(os.path.basename(filepath), encoding='utf-8'),
                            os.stat(filepath).st_size)  # 将xxx.jpg以128sq的格式打包

        s.send(fhead)
        fp = open(filepath, 'rb')  # 打开要传输的图片
        while True:
            data = fp.read(1024)  # 读入图片数据
            if not data:
                print('{0} send over...'.format(filepath))
                break
            s.send(data)  # 以二进制格式发送图片数据

        # 收图片
        sock = s
        fileinfo_size = struct.calcsize('128sq')
        buf = sock.recv(fileinfo_size)  # 接收图片名
        if buf:
            filename, filesize = struct.unpack('128sq', buf)
            fn = filename.decode().strip('\x00')
            new_filename = os.path.join('./',
                                        'new_' + fn)  # 在服务器端新建图片名（可以不用新建的，直接用原来的也行，只要客户端和服务器不是同一个系统或接收到的图片和原图片不在一个文件夹下）

            recvd_size = 0
            fp = open(new_filename, 'wb')

            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = sock.recv(1024)
                    recvd_size += len(data)
                else:
                    data = sock.recv(1024)
                    recvd_size = filesize
                fp.write(data)  # 写入图片数据
            fp.close()

        # 展示图片

        s.close()

        image_raw = cv2.imread(new_file)
        max_len = max(image_raw.shape[0], image_raw.shape[1])
        print(image_raw.shape)
        resized_image = cv2.resize(image_raw, (600, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(new_file, resized_image)
        pixmax_1 = QPixmap(new_file)

        self.label_1.setPixmap(pixmax_1)

    # def get_result(src_path):

    def saveFile(self):
        print('save--file')

    def load_text(self):
        print("load--text")
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFilter(QDir.Files)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            f = open(filenames[0], 'r')
            with f:
                data = f.read()
                self.content.setText(data)


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.InitUI()

    def InitUI(self):
        self.statusBar().showMessage('准备就绪')

        # linux上
        # self.setGeometry(500, 500, 600, 600)
        # window上
        self.setGeometry(250, 250, 1200, 600)
        self.setWindowTitle('优图速算识别')

        exitAct = QAction(QIcon('exit.png'), '退出(&E)', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('退出程序')
        exitAct.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('文件(&F)')
        fileMenu.addAction(exitAct)
        self.centeralwidget = filedialogdemo()
        self.setCentralWidget(self.centeralwidget)

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())