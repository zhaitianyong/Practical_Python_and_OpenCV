import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from day06.morphology import *


def Binary(img, threshold):
    temp = img.copy()
    temp[img<=threshold] = 0
    temp[img> threshold] = 255
    return temp

def FindConnected(img):
    # Read image
    img = img.astype(np.float32)
    H, W, C = img.shape

    label = np.zeros((H, W), dtype=np.int)
    label[img[..., 0] > 0] = 1 # 首先都标记为1



    LUT = [0 for _ in range(H * W)]

    n = 1

    for y in range(H):
        for x in range(W):
            if label[y, x] == 0:
                continue
            c3 = label[max(y - 1, 0), x]
            c5 = label[y, max(x - 1, 0)]
            if c3 < 2 and c5 < 2:
                n += 1
                label[y, x] = n
            else:
                _vs = [c3, c5]
                vs = [a for a in _vs if a > 1]
                v = min(vs)
                label[y, x] = v

                # 如何理解呢
                minv = v
                for _v in vs:
                    if LUT[_v] != 0:
                        minv = min(minv, LUT[_v])
                for _v in vs:
                    LUT[_v] = minv

    # 默认晒选出7个区域
    for i in range(1, n+1):
        idx = np.where(label==i)
        print(len(idx[0]))

    count = 1
    for l in range(2, n + 1):
        flag = True
        for i in range(n + 1):
            if LUT[i] == l:
                if flag:
                    count += 1
                    flag = False
                LUT[i] = count

    COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    out = np.zeros((H, W, C), dtype=np.uint8)

    for i, lut in enumerate(LUT[2:]):
        out[label == (i + 2)] = COLORS[lut - 2]

    return out



if __name__=="__main__":
    image = cv2.imread("../data/day06/seg.png")
    #gray = image[:,:, 0]
    #out = Gradient(binary)
    #binary = Binary(gray, 100)
    out = FindConnected(image)
    cv2.imshow("out", out)
    cv2.waitKey()