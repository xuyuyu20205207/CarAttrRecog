from flask import Flask, render_template,request
from flask_cors import *
import base64
import cv2
import numpy as np
from infer import recog_attr

app = Flask(__name__)


def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    return img


@app.route("/detect", methods=['POST'])
def detect():
    img = request.get_json(silent=True)
    element = img["img"].split('base64,')[-1]
    img = base64_to_image(element) #转码
    # print(img)
    new_filename = "web_1.jpg"
    cv2.imwrite(new_filename, img) #保存原图片

    img_vis, id, attr = recog_attr(img_path=new_filename) #识别
    cv2.imwrite("web_1_result.jpg", img_vis) #保存识别后的图片

    image = cv2.imencode('.jpg', img_vis)[1] #转码
    base64_data_str = str(base64.b64encode(image))[2:-1]
    html_str = "data:image/jpg;base64," + base64_data_str

    return html_str


if __name__ == '__main__':
    CORS(app, supports_credentials = True)
    app.run(host='0.0.0.0', port=5000, debug=True)