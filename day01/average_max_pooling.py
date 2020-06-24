
'''
  池化
  8*8 平均池化 的窗口
  问题七：平均池化（Average Pooling）
  问题八：最大池化（Max Pooling）

'''
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

sampel_data_path = os.path.join("../data/day01/")

print(sampel_data_path)

image = cv2.imread(sampel_data_path + "lena.jpg", cv2.IMREAD_COLOR)

h, w = image.shape[:2]
wSize = 8
size = (int(h/wSize), int(w/wSize), 3)
out_mean = np.zeros(size, dtype=np.uint8) # 平均池化
out_max = np.zeros(size, dtype=np.uint8) # 最大池化
for i in range(size[0]):
    for j in range(size[1]):
        block = image[i*wSize:i*wSize+wSize, j*wSize:j*wSize+wSize, :]
        for c in range(3):
            mean = int(np.mean(block[:,:,c]))
            max = block[:,:,c].max()
            out_mean[i, j, c] = mean
            out_max[i, j, c] = max

join_image = np.hstack((out_mean, out_max))

# cv2.imshow("out_mean", out_mean)
# cv2.imshow("out_max", out_max)
# cv2.waitKey()
plt.imshow(join_image)
plt.show()

