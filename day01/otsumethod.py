
# ostu method 大津算法 自适应算

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

sampel_data_path = os.path.join("../data/day01/")

print(sampel_data_path)

image = cv2.imread(sampel_data_path + "lena.jpg", cv2.IMREAD_GRAYSCALE)

threshold = 150

binary = image.copy()
h, w = binary.shape[0], binary.shape[1]


# 统计每个像素的个数
pixValues = [0 for i in range(256)]

for r in range(h):
    for c in range(w):
        v = image[r, c]
        pixValues[v] += 1

x = np.arange(0, 256, 1)

y = np.array(pixValues, dtype=float) # 每个像素的直方图
y_normal = y / (w*h) # 归一化的直方图比例



glist = np.zeros((256), dtype=int) # 类间方差数组
u = np.sum(image)/np.sum(y)  # 整体的像素平均值
for i in range(0, 256):
    w0 = np.sum(y_normal[:i+1])
    w1 = 1- w0
    n0 = np.sum(y[:i+1])
    n1 = np.sum(y[i + 1:])
    u0 = u1 = 0
    if n0 != 0:
        u0 = np.sum(y[:i+1].dot(x[:i+1])) / n0
    if n1 != 0:
        u1 = np.sum(y[i+1:].dot(x[i+1:])) / n1
    #print(w0, w1, u0, u1)

    g = w0*(u0-u)*(u0-u) + w1*(u1-u)*(u1-u) # 类间方差
    glist[i] = g


# 取最大的作为分割阈值
threshold = np.argmax(glist)

print(threshold)

binary = image.copy()

for r in range(h):
    for c in range(w):
        if image[r, c]< threshold:
            binary[r, c] = 0
        else:
            binary[r, c] = 255

# 显示
plt.subplot(2,2,1)
plt.bar(x, y_normal)
plt.subplot(2,2,2)
plt.plot(x, glist)
plt.subplot(2,2,3)
plt.imshow(image)
plt.subplot(2,2,4)
plt.imshow(binary)
plt.show()

