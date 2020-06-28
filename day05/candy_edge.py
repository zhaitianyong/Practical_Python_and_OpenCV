import cv2
import os
import numpy as np
import math

Pi = 3.141592653

def Candy(img):
    # 灰度化
    def bgr2gray( image):
        b = image[:, :, 0].copy()
        g = image[:, :, 1].copy()
        r = image[:, :, 2].copy()
        result = 0.0722 * b + 0.7152 * g + 0.2126 * r
        return result.astype(np.uint8)


    # 高斯滤波
    def gauss_filter( image, k=3, sigma=0.849):
        def gauss(x, y, sigma):
            sigma_2 = sigma * sigma
            A = 1 / (2 * Pi * sigma_2)
            return A * math.exp(-(x * x + y * y) / (2 * sigma_2))

        def createTemplate(k, sigma):
            kernel_ = np.zeros((k, k))
            N = int(k / 2)
            for i in range(-N, N + 1):
                for j in range(-N, N + 1):
                    kernel_[i+N, j+N] = gauss(i, j, sigma)
            kernel_ /= np.sum(kernel_)
            return kernel_

        kernel = createTemplate(k,sigma)

        N = int(k/2)
        padding = np.pad(image, (N, N), "constant")
        out = image.copy()
        h, w = image.shape
        for row in range(h):
            for col in range(w):
                block = padding[row: row+2*N + 1, col: col+2*N +1]
                sum = 0
                for i in range(k):
                    sum += block[i, :].dot(kernel[i, :])
                out[row, col] = int(sum)
        return out

    # sobel 算子
    def sobel( image, axis=0):
        if axis ==0: # 横向 x 方向
            kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float)
        else:
            kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)

        k = 3
        N = k // 2
        padding = np.pad(image, (N, N), "constant")
        out = np.zeros(image.shape, dtype=np.float)
        h, w = image.shape
        for row in range(h):
            for col in range(w):
                block = padding[row: row + 2 * N + 1, col: col + 2 * N + 1]
                sum = 0
                for i in range(k):
                    sum += block[i, :].dot(kernel[i, :])
                out[row, col] = sum
        return out

    # 计算角度
    def angle( sobel_x, sobel_y):
        sobel_x[sobel_x == 0] = 0.0001
        edge_tan = np.arctan(sobel_y / sobel_x)
        edge_angle = np.zeros(sobel_x.shape, np.int)
        edge_angle[np.where((edge_tan>-0.4142) & (edge_tan<= 0.4142))]  = 0
        edge_angle[np.where((edge_tan > 0.4142) & (edge_tan < 2.4142))] = 45
        edge_angle[np.where((edge_tan >= 2.4142) | (edge_tan <= -2.4142))] = 90
        edge_angle[np.where((edge_tan > -2.4142) & (edge_tan <= -0.4142))] = 45
        return edge_angle

    # 非极大值抑制（边缘细化）
    def nms( edge, edge_angle):
        edge_nms = edge.copy()
        h, w = edge.shape
        for row in range(1, h - 1):
            for col in range(1, w - 1):
                ang = edge_angle[row, col]
                ed = edge[row, col]
                if ang == 0:
                    if ed < edge[row, col - 1] or ed < edge[row, col + 1]:
                        edge_nms[row, col] = 0
                elif ang == 45:
                    if ed < edge[row - 1, col + 1] or ed < edge[row + 1, col - 1]:
                        edge_nms[row, col] = 0
                elif ang == 90:
                    if ed < edge[row - 1, col] or ed < edge[row + 1, col]:
                        edge_nms[row, col] = 0
                elif ang == 135:
                    if ed < edge[row - 1, col - 1] or ed < edge[row + 1, col + 1]:
                        edge_nms[row, col] = 0
        return edge_nms

    # 二值化
    def thresh( edge, lt, ht):
        #edge[edge>ht] = 255
        #edge[edge<lt] = 0
        out = edge.copy()
        h, w = edge.shape
        for row in range(1, h-1):
            for col in range(1, w-1):
                dxy = edge[row, col]
                if dxy <= lt:
                    out[row, col] = 0
                elif dxy >= ht:
                    out[row, col] = 255
                else:
                    block = edge[row-1: row+2, col-1: col+2]
                    block[1,1] = 0  # 把中心的值设为0 计算其他8个邻域的最大值
                    if dxy > np.max(block):
                        out[row, col] = 255
                    else:
                        out[row, col] = 0
        return out


    # 灰度
    gray = bgr2gray(img)
    # 高斯平滑
    gauss = gauss_filter(gray, k=5, sigma=1.4)
    # sobel 算子
    sobel_x = sobel(gauss, 0)
    sobel_y = sobel(gauss, 1)
    # 边缘和角度
    edge = 0.5 * np.abs(sobel_y) + 0.5 * np.abs(sobel_y)
    edge_angle = angle(sobel_x, sobel_y)
    # 边缘细化
    edge_nms = nms(edge, edge_angle)
    # 滞后阈值r
    edge_thresh = thresh(edge_nms, 30, 100)
    out = edge_thresh.astype(np.uint8)
    return out


def Hough_Line_step1(edge, img):
    ## Voting
    def voting(edge):
        H, W = edge.shape
        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

        # hough table
        hough = np.zeros((rho_max*2, 180), dtype=np.int)

        # get index of edge
        ind = np.where(edge == 255)

        ## hough transformation
        for y, x in zip(ind[0], ind[1]):
            for theta in range(0, 180, dtheta):
                # get polar coordinat4s
                t = np.pi / 180 * theta
                rho = int(x * np.cos(t) + y * np.sin(t))

                #print(rho+rho_max)
                # vote
                hough[rho + rho_max, theta] += 1
                #hough[rho, theta] += 1

        out = hough.astype(np.uint8)

        return out

    # non maximum suppression
    def non_maximum_suppression(hough):
        rho_max, _ = hough.shape

        ## non maximum suppression
        for y in range(rho_max):
            for x in range(180):
                # get 8 nearest neighbor
                x1 = max(x - 1, 0)
                x2 = min(x + 2, 180)
                y1 = max(y - 1, 0)
                y2 = min(y + 2, rho_max - 1)
                if np.max(hough[y1:y2, x1:x2]) == hough[y, x] and hough[y, x] != 0:
                    pass
                # hough[y,x] = 255
                else:
                    hough[y, x] = 0

        return hough

    def inverse_hough(hough, img):
        H, W, _ = img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180 - rho_max / 2

        # each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta

            # hough -> (x,y)
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out[y, x] = [0, 0, 255]
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [0, 0, 255]

        out = out.astype(np.uint8)

        return out

    # voting
    hough = voting(edge)

    # non maximum suppression
    hough = non_maximum_suppression(hough)

    # inverse hough
    out = inverse_hough(hough, img)

    return out

if __name__ == "__main__":
    image = cv2.imread("../data/day05/thorino.jpg")
    candy = Candy(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    out = Hough_Line_step1(candy, image)
    cv2.imshow("out",out)
    cv2.waitKey()


