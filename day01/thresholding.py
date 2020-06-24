#
# 二值化，转换为灰度图，然后根据阈值，设为0， 255

import os
import cv2
import numpy as np

sampel_data_path = os.path.join("../data/day01/")

print(sampel_data_path)

image = cv2.imread(sampel_data_path + "lena.jpg", cv2.IMREAD_GRAYSCALE)

threshold = 150

binary = image.copy()

h, w = binary.shape[0], binary.shape[1]

for r in range(h):
    for c in range(w):
        if(binary[r, c] < threshold):
            binary[r, c] = 0
        else:
            binary[r, c] = 255

#binary[:, :]

cv2.imshow("out", binary)
cv2.waitKey()