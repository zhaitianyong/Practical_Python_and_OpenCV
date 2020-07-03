import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def BGR2GRAY(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    gray = 0.0722 * b + 0.7152 * g + 0.2126 * r
    return gray.astype(np.uint8)


def OSTU(img):
    # 最大类内方差
    # w1, w2  u1, u2  u
    # w1*(u1 - u)*(u1-u) + w2*(u2-u)*(u2-u)

    def getThreshold(img):
        # hist = np.zeros((256), np.int)
        h, w = img.shape
        # for i in range(256):
        #     hist[i] = len(np.where(img == i))
        total = h * w
        u = np.mean(img)
        maxIndex, maxValue = 0, 0
        for i in range(256):
            indx = np.where(img <= i)
            indy = np.where(img > i)
            if len(indx[0]) == 0:
                w1, w2, u1, u2 = 0., 1., 0., u
            elif len(indy[0]) == 0:
                w1, w2, u1, u2 = 1., 0., u, 0.
            else:
                w1 = len(indx[0]) / total
                w2 = 1 - w1
                u1 = np.mean(img[indx])
                u2 = np.mean(img[indy])
            g = w1 * math.pow(u1 - u, 2) + w2 * math.pow(u2 - u, 2)
            if maxValue < g:
                maxValue = g
                maxIndex = i
        print(maxIndex, maxValue)
        return maxIndex

    def binary(img, threshold):
        img[img<threshold] = 0
        img[img>= threshold] = 255
        return img

    temp = img.copy()

    threshold = getThreshold(temp)

    out = binary(temp, threshold)
    return out


def Dilate(img, iters=1):
    out = img.copy()
    h, w = img.shape
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    temp = img.copy()
    for i in range(iters):
        indx = np.where(temp == 0)
        for r, c in zip(indx[0], indx[1]):
            min_r = max(r-1, 0)
            min_c = max(c-1, 0)
            max_r = min(r+1, h)
            max_c = min(c+1, w)
            block = temp[min_r:max_r+1, min_c: max_c+1]
            if block.shape[0] == 3 and block.shape[1] == 3:
                if np.max(kernel*block) == 255:
                    out[r, c] = 255
        temp = out.copy()
    return out

def Erode(img, iters=1):
    out = img.copy()
    h, w = img.shape
    temp = img.copy()
    for i in range(iters):
        indx = np.where(temp == 255)
        for r, c in zip(indx[0], indx[1]):
            min_r = max(r - 1, 0)
            min_c = max(c - 1, 0)
            max_r = min(r + 1, h)
            max_c = min(c + 1, w)
            block = temp[min_r:max_r + 1, min_c: max_c + 1]
            if block.shape==(3, 3):
                value =np.array([block[0, 1], block[1, 0], block[1, 1], block[2, 1]])
                if np.min(value) == 0:
                    out[r, c] = 0
        temp = out.copy()

    return out

# 开运行 ，先腐蚀 - 再膨胀， 去除白色噪点
def Opening(img, iters=1):
    di = Erode(img, iters)
    er = Dilate(di, iters)
    return er

# 闭运算， 先膨胀，再腐蚀， 去除黑色噪点
def Closing(img, iters=1):
    er = Dilate(img, iters)
    di = Erode(er, iters)
    return di

# 形态学梯度为经过膨胀操作（dilate）的图像与经过腐蚀操作（erode）的图像的差，可以用于抽出物体的边缘
def Gradient(img):
    di = Dilate(img)
    er = Erode(img)
    g = di - er
    return  g

# 顶帽运算是原图像与开运算的结果图的差
def TopHat(img):
    open = Opening(img)
    return img - open

# 黑帽运算是原图像与闭运算的结果图的差
def BlakHat(img):
    close = Closing(img)
    return img - close


if __name__ == "__main__":
    image = cv2.imread("../data/day05/imori.jpg")
    gray = BGR2GRAY(image)
    # cv2.imshow("out", gray)
    # cv2.waitKey()

    binary = OSTU(gray)

    out = Erode(binary)
    # out2 = erode(binary)
    # out = TopHat(binary)
    result = np.hstack((binary, out))
    cv2.imshow("out", result)
    cv2.waitKey()
    # kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    #
    # temp = np.arange(1, 10).reshape((3, 3))
    # print(kernel)
    # print(temp)
    # print(temp*kernel)