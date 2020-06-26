'''
高斯滤波
高斯滤波（Gaussian Filter）

'''

import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

sample_data_path = os.path.join("../data/day01/")
PI = 3.1415926
print(sample_data_path)


def gaussian(i, j, sigma=1.3):
    sigma2 = math.pow(sigma, 2)
    g = math.exp(-(math.pow(i - 1, 2) + math.pow(j - 1, 2)) / (2 * sigma2))
    return g / (2 * PI * sigma2)


# 创建滤波器
def createFilter(w=5):
    k = int(w / 2)
    template = np.zeros((w, w), dtype=np.float)
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            template[i, j] = gaussian(i, j)
    sum = np.sum(template)
    template /= sum
    return template


image = cv2.imread(sample_data_path + "lena.jpg", cv2.IMREAD_COLOR)

h, w, c = image.shape

print(image.shape)

# 卷积计算
wSize = 3
k = int(wSize / 2)
template = createFilter(wSize )
print(np.sum(template))

filter_image = np.zeros(image.shape, dtype=image.dtype)

for row in range(k, h - k):
    for col in range(k, w - k):
        for ch in range(c):
            block = image[row - k:row + k + 1, col - k:col + k + 1, ch]
            # print(row, col , ch, np.mean(block))
            sum = 0
            for i in range(wSize):
               sum += block[i, :].dot(template[i, :])
            filter_image[row, col, ch] = int(sum)

cv2.imshow("out", filter_image)
cv2.waitKey()
