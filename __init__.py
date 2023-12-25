import csv
import json
import os

import numpy as np
from PIL import Image
from flask import Flask, render_template, send_from_directory, request, Response

from flask_cors import *

import cv2 as cv

from matplotlib import pyplot as plt
from skimage import measure, filters, img_as_ubyte

from strUtil import pic_str

from fcm import get_centroids, get_label, get_init_fuzzy_mat, fcm
from lv_set import find_lsf, get_params

from preprocess import gamma_trans, clahe_trans

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50

from src import UNet

from src import deeplabv3_resnet50

from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN

from draw_box_utils_mg import draw_objs

# 配置Flask路由，使得前端可以访问服务器中的静态资源
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'gif', 'tif'}

global src_img, pic_path, res_pic_path, message_get, pic_name, final


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def del_files(dir_path):
    # os.walk会得到dir_path下各个后代文件夹和其中的文件的三元组列表，顺序自内而外排列，
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        # 第二步：删除空文件夹
        for name in dirs:
            os.rmdir(os.path.join(root, name))  # 删除一个空目录


# 主页
@app.route('/')
def hello():
    return render_template('main.html')


@app.errorhandler(404)
def miss404(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def miss500(e):
    return render_template('errors/500.html'), 500


@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/example/')
def example_index():
    return render_template('example/example-datasets.html')


@app.route('/example/classic')
def example_classic_index():
    return render_template('example/classic/index.html')


# 展示海马体exp
@app.route('/example/classic/hippocampus')
def show_example_hippocampus():
    filename = 'static/csv/hippocampus_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
    return render_template('example/classic/hippocampus-index.html', header=header, data1=data1, data2=data2,
                           data3=data3, data4=data4)


# 展示胸部X光exp
@app.route('/example/classic/chest')
def show_example_chest():
    filename = 'static/csv/chest_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
    return render_template('example/classic/chest-index.html', header=header, data1=data1, data2=data2, data3=data3,
                           data4=data4, data5=data5, data6=data6)


# 展示眼底血管exp
@app.route('/example/classic/eye')
def show_example_eye():
    filename = 'static/csv/eye_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
    return render_template('example/classic/eye-index.html', header=header, data1=data1, data2=data2, data3=data3,
                           data4=data4, data5=data5, data6=data6)


@app.route('/index_data_chest')
def line_stack_data_chest():
    data_list = {}
    filename = 'static/csv/chest_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7

    return Response(json.dumps(data_list), mimetype='application/json')


# eye_exp_data
@app.route('/index_data_eye')
def data_eye():
    data_list = {}
    filename = 'static/csv/eye_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7

    return Response(json.dumps(data_list), mimetype='application/json')


# hippocampus_exp_data
@app.route('/index_data_hippocampus')
def data_hippocampus():
    data_list = {}
    filename = 'static/csv/hippocampus_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    return Response(json.dumps(data_list), mimetype='application/json')


# 展示不同网络对睑板腺数据集的分割效果
@app.route("/example/dl")
def dl_data():
    filename = 'static/csv/dl_data.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
    return render_template('example/dl/dl-index.html', data1=data1, data2=data2, data3=data3)


@app.route("/dl_data1")
def dl_data_m():
    data_list = {}
    filename = 'static/csv/dl_data1.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
    data_list['header'] = header
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route("/dl_data2")
def dl_data_c():
    data_list = {}
    filename = 'static/csv/dl_data2.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
    data_list['header'] = header
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    return Response(json.dumps(data_list), mimetype='application/json')


# 实时体验功能首页
@app.route('/live-index/')
def live_index():
    return render_template('/live/live-index.html')


# 图片上传相关
@app.route('/live-classic')
def upload_test():
    global pic_path, res_pic_path
    pic_path = 'assets/img/svg/illustration-3.svg'
    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-classic.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route('/live-classic/upload-success', methods=['POST'])
def upload_pic():
    del_files('static/tempPics')
    img1 = request.files['photo']
    if img1 and allowed_file(img1.filename):
        img = Image.open(img1.stream)

    # 保存图片
    global pic_path, res_pic_path
    # 为临时图片生成随机id
    pic_path = 'tempPics/' + pic_str().create_uuid() + '.png'
    img.save('static/' + pic_path)
    global src_img
    src_img = cv.imread('static/' + pic_path)
    src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    # 预处理
    src_img = clahe_trans(src_img)
    src_img = gamma_trans(src_img)

    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-classic.html', pic_path=pic_path, res_pic_path=res_pic_path)


# 获取算法信息
@app.route('/live-classic/upload-success', methods=['GET'])
def get_Algorithm():
    global message_get
    message_get = str(request.values.get("algorithm"))


# 使用对应算法进行处理
@app.route('/live-classic/upload-success/result')
def algorithm_process():
    global src_img, res_pic_path, pic_path, message_get, pic_name
    if message_get == 'SOBEL':
        # 边缘检测之Sobel 算子
        edges = filters.sobel(src_img)
        # 浮点型转成uint8型
        edges = img_as_ubyte(edges)
        plt.figure()
        plt.imshow(edges, plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        pic_name = 'sobel.png'
        res_pic_path = 'tempPics/' + pic_name
        plt.savefig('static/' + res_pic_path)

    elif message_get == 'OTSU':
        _, otsu_img = cv.threshold(src_img, 0, 255, cv.THRESH_OTSU)
        pic_name = 'eye_otsu.png'
        res_pic_path = 'tempPics/' + pic_name
        cv.imwrite('static/' + res_pic_path, otsu_img)

    elif message_get == 'WATERSHED':
        # 基于直方图的二值化处理
        _, thresh = cv.threshold(src_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # 做开操作，是为了除去白噪声
        kernel = np.ones((3, 3), dtype=np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

        # 做膨胀操作，是为了让前景漫延到背景，让确定的背景出现
        sure_bg = cv.dilate(opening, kernel, iterations=2)

        # 为了求得确定的前景，也就是注水处使用距离的方法转化
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        # 归一化所求的距离转换，转化范围是[0, 1]
        cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)
        # 再次做二值化，得到确定的前景
        _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 得到不确定区域也就是边界所在区域，用确定的背景图减去确定的前景图
        unknow = cv.subtract(sure_bg, sure_fg)

        # 给确定的注水位置进行标上标签，背景图标为0，其他的区域由1开始按顺序进行标
        _, markers = cv.connectedComponents(sure_fg)

        # 让标签加1，这是因为在分水岭算法中，会将标签为0的区域当作边界区域（不确定区域）
        markers += 1

        # 是上面所求的不确定区域标上0
        markers[unknow == 255] = 0

        # 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
        src_img = cv.cvtColor(src_img, cv.COLOR_GRAY2BGR)
        markers = cv.watershed(src_img, markers)

        # 分水岭算法得到的边界点的像素值为-1
        src_img[markers == -1] = [0, 0, 255]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(src_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        pic_name = 'watershed.png'
        res_pic_path = 'tempPics/' + pic_name
        fig.savefig('static/' + res_pic_path, bbox_inches='tight')

    elif message_get == 'FCM':
        rows, cols = src_img.shape[:2]
        pixel_count = rows * cols
        image_array = src_img.reshape(1, pixel_count)

        # 初始模糊矩阵
        init_fuzzy_mat = get_init_fuzzy_mat(pixel_count)
        # 初始聚类中心
        init_centroids = get_centroids(image_array, init_fuzzy_mat)
        fuzzy_mat, centroids, target_function = fcm(init_fuzzy_mat, init_centroids, image_array)
        label = get_label(fuzzy_mat, image_array)
        fcm_img = label.reshape(rows, cols)
        pic_name = 'fcm.png'
        res_pic_path = 'tempPics/' + pic_name
        cv.imwrite('static/' + res_pic_path, fcm_img)

    elif message_get == 'DRLSE':
        global final
        final = 0
        src_img = cv.resize(src_img, (128, 128))
        params = get_params(src_img)
        phi = find_lsf(**params)

        contours = measure.find_contours(phi, 0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(src_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            final = contour

        ax.fill(final[:, 1], final[:, 0], color='w')
        ax.set_xticks([])
        ax.set_yticks([])

        pic_name = 'drlse.png'
        res_pic_path = 'tempPics/' + pic_name
        print(res_pic_path)
        fig.savefig('static/' + res_pic_path, bbox_inches='tight')

        del params

    return render_template('live/show.html', pic_path=pic_path, res_pic_path=res_pic_path, temp=message_get)


# 图片下载
@app.route('/live-classic/upload-success/result/download', methods=['GET'])
def download():
    global res_pic_path
    if request.method == "GET":
        path = 'static/tempPics'
        if path:
            return send_from_directory(path, pic_name, as_attachment=True)


# 获取dl算法信息
@app.route('/live-dl')
def upload_test1():
    global pic_path, res_pic_path
    pic_path = 'assets/img/svg/illustration-3.svg'
    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-dl.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route('/live-dl/upload-success', methods=['POST'])
def upload_pic1():
    del_files('static/tempPics')
    img1 = request.files['photo']
    if img1 and allowed_file(img1.filename):
        img = Image.open(img1.stream)
    # 保存图片
    global pic_path, res_pic_path
    # 为临时图片生成随机id
    pic_path = 'tempPics/' + pic_str().create_uuid() + '.png'
    img.save('static/' + pic_path)

    res_pic_path = 'assets/img/svg/illustration-7.svg'
    return render_template('live/live-dl.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route('/live-dl/upload-success', methods=['GET'])
def get_Algorithm1():
    global message_get
    message_get = str(request.values.get("algorithm"))


@app.route('/live-dl/upload-success/result')
def algorithm_dl():
    global pic_path, message_get, pic_name, res_pic_path
    palette_path = 'static/assets/weights/palette.json'

    weights_path = 'static/assets/weights/' + message_get + '_best_model.pth'
    img_path = 'static/' + pic_path
    pic_name = message_get + '.png'
    num_classes = 3
    print(message_get)
    with open(palette_path, "rb") as f:
        palette_dict = json.load(f)
        palette = []
        for v in palette_dict.values():
            palette += v
    aux = False  # inference time not need aux_classifier

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if message_get == 'fcn':
        # create model
        model = fcn_resnet50(aux=aux, num_classes=num_classes)
        # delete weights about aux_classifier
        weights_dict = torch.load(weights_path, map_location='cpu')['model']
        for k in list(weights_dict.keys()):
            if "aux" in k:
                del weights_dict[k]

        # load weights
        model.load_state_dict(weights_dict)
        model.to(device)

        # load image
        original_img = Image.open(img_path)

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(420),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(palette)

    elif message_get == 'unet':
        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # create model
        model = UNet(in_channels=3, num_classes=num_classes, base_c=32)

        # load weights
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.to(device)

        # load image
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            # 预测多目标
            mask = Image.fromarray(prediction)
            mask.putpalette(palette)

    elif message_get == 'deeplab':
        # create model
        model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)

        # delete weights about aux_classifier
        weights_dict = torch.load(weights_path, map_location='cpu')['model']
        for k in list(weights_dict.keys()):
            if "aux" in k:
                del weights_dict[k]

        # load weights
        model.load_state_dict(weights_dict)
        model.to(device)

        # load image
        original_img = Image.open(img_path)

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(420),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            output = model(img.to(device))
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(palette)

            # out = np.array(original_img) * 0.5 + np.array(mask) * 0.5

    elif message_get == 'maskrcnn':
        # create model
        model = create_model(num_classes=num_classes, box_thresh=0.5)
        label_json_path = 'static/assets/weights/mg2_indices.json'

        # load train weights
        assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
        weights_dict = torch.load(weights_path, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        model.load_state_dict(weights_dict)
        model.to(device)

        # read class_indict
        assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
        with open(label_json_path, 'r') as json_file:
            category_index = json.load(json_file)

        # load image
        assert os.path.exists(img_path), f"{img_path} does not exits."
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            predictions = model(img.to(device))[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                return

            mask = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
            plt.show()

    # 保存预测的图片结果
    res_pic_path = 'tempPics/' + pic_name
    mask.save('static/' + res_pic_path)
    return render_template('live/show-dl.html', pic_path=pic_path, res_pic_path=res_pic_path, temp=message_get)


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


# 图片下载
@app.route('/live-dl/upload-success/result/download', methods=['GET'])
def download1():
    global res_pic_path
    if request.method == "GET":
        path = 'static/tempPics'
        if path:
            return send_from_directory(path, pic_name, as_attachment=True)


if __name__ == '__main__':
    app.run(port=205, debug=True)
