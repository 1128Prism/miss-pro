import math

import h5py
import numpy as np
from PIL import Image
import cv2


def gamma_trans(img):  # gamma函数处理
    mean = np.mean(img)
    gamma = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


# create a CLAHE object (Arguments are optional).
def clahe_trans(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img)
    return clahe_image
