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


def gaussian(i, j, sigma=0.8):
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

    return template


image = cv2.imread(sample_data_path + "lena.jpg", cv2.IMREAD_COLOR)

h, w, c = image.shape

print(image.shape)

# 卷积计算
w = 3
k = int(w / 2)
template = createFilter(w)
print(template)

filter_image = np.zeros(image.shape, dtype=image.dtype)
for row in range(k, h - k, 1):
    for col in range(k, w - k, 1):
        for ch in range(c):
            block = image[row - k:row + k + 1, col - k:col + k + 1, ch]
            # print(block)
            #print(block.shape)
            #if (block.shape[0] != w or block.shape[1] != w):
             #   continue
            sum = 0
            for i in range(w):
                for j in range(w):
                    sum += block[i, j] * template[i, j]

            filter_image[row, col, ch] = int(sum)
print(filter_image)

plt.imshow(filter_image)
plt.show()
