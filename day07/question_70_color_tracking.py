'''
 色彩跟踪

 color tracking
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from day06.morphology import Opening, Closing


# HSV - Hue, Saturation, Value
# brg 2 hsv
def BGR2HSV(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    normal = img / 255
    c_max = np.max(normal, axis=2)
    c_min = np.min(normal, axis=2)

    print(c_max.shape, c_min.shape)
    out = np.zeros(img.shape, dtype=np.float)

    H, W, C = img.shape

    for row in range(H):
        for col in range(W):
            pix = normal[row, col, :]

            dx = c_max[row, col] - c_min[row, col]
            # hue
            if dx == 0:
                pass
            elif c_min[row, col] == pix[0]:
                out[row, col, 0] = (pix[1] - pix[2]) / dx * 60 + 60
            elif c_min[row, col] == pix[2]:
                out[row, col, 0] = (pix[0] - pix[1]) / dx * 60 + 180
            elif c_min[row, col] == pix[1]:
                out[row, col, 0] = (pix[2] - pix[0]) / dx * 60 + 300

            # saturation
            out[row, col, 1] = dx

            # value

            out[row, col, 2] = c_max[row, col]

    return out


def Show(img):
    cv2.imshow("out", img)
    cv2.waitKey()


# 问题七十：色彩追踪（Color Tracking）
def ColorTrack(img):
    hsv = BGR2HSV(img)

    H = hsv[..., 0]
    h, w, c = img.shape
    out = np.zeros((h, w), dtype=np.uint8)
    idx = np.where((H >= 180) & (H <= 260))
    out[idx] = 255
    return out


def Masking(img):
    mask = ColorTrack(img)
    # 取反
    # 形态学处理Mask 数据
    # 开操作和闭操作
    # 开操作 先腐蚀运算，再膨胀运算 去除白色噪点
    Show(mask)
    mask = Opening(mask, 5)
    Show(mask)

    mask = Closing(mask, 5)
    Show(mask)
    # 设置为0， 1
    mask[mask == 0] = 1
    mask[mask == 255] = 0

    mask = np.dstack((mask, mask, mask))
    out = img * mask

    return out


# 重新放大缩小
def Resize(img, size, mode=0):
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1

    H_ = size[0]
    W_ = size[1]

    if C==1:
        out = np.zeros((H_, W_), dtype=np.uint8)
    else:
        out = np.zeros((H_, W_, C), dtype=np.uint8)

    for y_ in range(H_):
        for x_ in range(W_):
            x = W / W_ * (x_+0.5) - 0.5
            y = H / H_ * (y_+0.5) - 0.5
            if mode == 0:
                pix = NearestNeighborInterpolation(x, y, img)
                out[y_, x_] = pix
            elif mode == 1:
                pix = BilinearInterpolation(x, y, img)
                out[y_, x_] = pix
    return out


def PryDown(img, r=0.5):
    H_ = int(img.shape[0]*r)
    W_ = int(img.shape[1]*r)

    out = Resize(img, (H_, W_), mode=1)
    return out

def PryUp(img, r=2):
    H_ = int(img.shape[0] * r)
    W_ = int(img.shape[1] * r)
    out = Resize(img, (H_, W_), mode=1)
    return out

def NearestNeighborInterpolation(x, y, img):
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1

    x_ = int(x + 0.5)
    y_ = int(y + 0.5)
    if C==1:
        return img[y_, x_]
    else:
        return img[y_, x_, :]


def BilinearInterpolation(x, y, img):
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1

    #
    x0 = int(x)
    y0 = int(y)
    if C==1:
        I00 = img[y0, x0]
        I01 = img[y0, min(x0 + 1, W-1)]
        I10 = img[min(y0 + 1, H-1), x0]
        I11 = img[min(y0 + 1, H-1), min(x0 + 1, W-1)]
    else:
        I00 = img[y0, x0, :]
        I01 = img[y0, min(x0 + 1, W - 1), :]
        I10 = img[min(y0 + 1, H - 1), x0, :]
        I11 = img[min(y0 + 1, H - 1), min(x0 + 1, W - 1), :]

    dx = x - x0
    dy = y - y0

    I = (1 - dx) * (1 - dy) * I00 + dx * (1 - dy) * I01 + (1 - dx) * dy * I10 + dx * dy * I11

    return I.astype(np.uint8)



def BGR2GRAY(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    gray = 0.0722 * b + 0.7152 * g + 0.2126 * r
    return gray.astype(np.uint8)

def Diff(img):
    gray = BGR2GRAY(img)
    #
    down = PryDown(gray)

    gray_= PryUp(down)

    print(gray.shape, gray_.shape)
    # out = Masking(img)

    diff = np.abs(gray - gray_)

    out = diff / diff.max() * 255

    return out.astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("../data/day05/imori.jpg")

    out = Diff(img)


    Show(out)
