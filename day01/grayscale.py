#
#  灰度化 利用三个通道的比例进行计算
#  $$ Y = 0.2126\ R + 0.7152\ G + 0.0722\ B $$

import os
import cv2
import numpy as np

sampel_data_path = os.path.join("../data/day01/")

print(sampel_data_path)

image = cv2.imread(sampel_data_path + "lena.jpg", cv2.IMREAD_COLOR)

result = 0.0722 * image[:, :, 0] + 0.7152 * image[:, :, 1] +  0.2126 * image[:, :, 2]

gray = result.astype(np.uint8)

print(gray.shape, gray.dtype)

cv2.imshow("out", gray)
cv2.waitKey()


