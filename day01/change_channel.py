# 交换通道
# 一般图像都是 RBG三个通道
# opencv 读取的图像的通道顺序为BGR
import os
import cv2
import numpy as np

sampel_data_path = os.path.join("../data/day01/")

print(sampel_data_path)
image = cv2.imread(sampel_data_path + "lena.jpg", cv2.IMREAD_COLOR)
print(image.shape)
# bgr
cv2.imshow("out", image)
cv2.waitKey()
# rgb
# 思路1 先把每个通道分开，再分别赋值 深拷贝
b = image[:, :, 0].copy()
g = image[:, :, 1].copy()
r = image[:, :, 2].copy()
# 创建新的图像
result = np.zeros(image.shape, dtype=image.dtype)
result[:,:, 0] = r
result[:,:, 1] = g
result[:,:, 2] = b

cv2.imshow("result", result)
cv2.waitKey()

