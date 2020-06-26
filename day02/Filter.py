
'''
 滤波的类

'''
import cv2
import numpy as np
import math
PI  = 3.1415926
class Filter(object):

    def __init__(self):
        pass

    def test(self):
        print("test function")

    # 边界补零
    def padding(self, image, k = 3):
        b = int(k/2)

        if len(image.shape)==2:
            h, w= image.shape
            size = (h + 2 * b, w + 2 * b)

        else:
            h, w, c = image.shape
            size = (h + 2 * b, w + 2 * b, c)
        out = np.zeros(size, dtype=image.dtype)
        out[b:b+h, b:b+w] = image.copy()
        return out


    # 问题十一：均值滤波器
    def meanFilter(self, image, k=3):
        # kernal = np.ones((k, k), dtype=float)
        # kernal /= np.sum(kernal)
        temp = self.padding(image, k)
        h, w, c = temp.shape
        #step = int (k / 2)
        out = image.copy()
        for row in range(h-k):
            for col in range(w-k):
                for ch in range(c):
                   block = temp[row:row+k, col:col+k, ch]
                   out[row, col, ch] = int(np.mean(block))
        return out


    '''
    问题十二：Motion Filter
    '''
    def motionFilter(self, image, k=3):
        temp = self.padding(image, k)
        h, w, c = temp.shape
        out = image.copy()
        for row in range(h-k):
            for col in range(w-k):
                for ch in range(c):
                   block = temp[row:row+k, col:col+k, ch]
                   sum = 0
                   for i in range(k):
                       sum += block[i,i]
                   out[row, col, ch] = int(sum /k)
        return out

    def minMaxFilter(self, image, k=3):
        temp = self.padding(image, k)
        h, w = temp.shape
        out = image.copy()
        for row in range(h-k):
            for col in range(w-k):
               block = temp[row:row+k, col:col+k]
               max = block.max()
               min = block.min()

               out[row, col] = int(max - min)
        return out


    def sobelFilter(self, image, axis=0):
        k=3
        if axis==0:
            kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.int)
        else:
            kernel = np.array([[1,0,-1], [2,0,-2],[1,0,-1]], dtype=np.int)
        temp = self.padding(image,3)
        h, w = temp.shape
        out = np.zeros(image.shape, dtype=np.float)
        for row in range(h - k):
            for col in range(w - k):
                block = temp[row:row + k, col:col + k]
                sum = 0
                for i in range(k):
                    sum += block[i,:].dot(kernel[i,:])
                out[row, col] = sum
        return out


    def prewittFilter(self, image, axis=0):
        k = 3
        if axis == 0:
            kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.int)
        else:
            kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.int)
        temp = self.padding(image, 3)
        h, w = temp.shape
        out = np.zeros(image.shape, dtype=np.float)
        for row in range(h - k):
            for col in range(w - k):
                block = temp[row:row + k, col:col + k]
                sum = 0
                for i in range(k):
                    sum += block[i, :].dot(kernel[i, :])
                out[row, col] = sum
        return out

    def laplacianFilter(self, image):
        k = 3
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.int)
        temp = self.padding(image, 3)
        h, w = temp.shape
        out = np.zeros(image.shape, dtype=image.dtype)
        for row in range(h - k):
            for col in range(w - k):
                block = temp[row:row + k, col:col + k]
                sum = 0
                for i in range(k):
                    sum += block[i, :].dot(kernel[i, :])
                out[row, col] = sum
        return out

    def embossFilter(self, image):
        k = 3
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.int)
        temp = self.padding(image, 3)
        h, w = temp.shape
        out = np.zeros(image.shape, dtype=image.dtype)
        for row in range(h - k):
            for col in range(w - k):
                block = temp[row:row + k, col:col + k]
                sum = 0
                for i in range(k):
                    sum += block[i, :].dot(kernel[i, :])
                out[row, col] = sum
        return out

    def log(self, x, y, s):
        g = math.exp(-(x*x+y*y)/(2*s*s))
        g *= (x*x + y*y-s*s)/(2*PI*math.pow(s, 6))
        return g


    '''
     拉普拉斯+高斯   可以先高斯降噪然后拉普拉斯边缘增强
     也可以整合到一起
    '''
    def logFilter(self, image):
        s = 1
        kernel = np.zeros((3,3), dtype=np.float)
        for i in range(-1, 2):
            for j in range(-1, 2):
                kernel[i,j] = self.log(i, j, s)
        print(kernel)
