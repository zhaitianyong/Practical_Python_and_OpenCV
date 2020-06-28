
import cv2
import numpy as np
import math
class Histogram(object):
    def __init__(self):
        pass

    def hist(self,image):
        bin = np.zeros((256, 1),dtype= np.int)
        h, w = image.shape
        for r in range(h):
            for c in range(w):
                value = image[r, c]
                bin[value] += 1
        return bin

    def normalize(self, image, a =0, b=255):
        c = image.min()
        d = image.max()
        out = image.copy()
        out = (b-a)/(d-c)*(out-c) + a
        out[out<a] = a
        out[out>b] = b
        out = out.astype(np.uint8)
        return out

    def opera(self,image,m0=128, s0=52):
        m = np.mean(image)
        s = np.std(image)

        out = image.copy()
        out = s0/s * (out-m)+m0
        out[out<0] = 0
        out[out>255] = 255
        out = out.astype(np.uint8)
        return out

    def equl(self, image, zmax=255):
        h, w, c = image.shape
        s= h*w*c

        out = image.copy()
        sum_h=0
        for i in range(1, 255):
            ind = np.where(image == i)
            sum_h += len(image[ind])
            pre = zmax/s * sum_h
            out[ind] = pre
        return out

    def gamma_correction(self, image,c=1, g=2.2):

        out = image.copy()

        norm = out / 255

        out = (1/c)*np.power(norm, 1/g)
        out = out*255

        out = out.astype(np.uint8)

        return out

    def resize_near(self, image, ratio=1.5):
        h, w, c = image.shape
        h_ = int(h*ratio)
        w_ = int(w*ratio)

        out = np.zeros((h_, w_, c), dtype=image.dtype)
        for row in range(h_):
            for col in range(w_):
                i = int(row / ratio)
                j = int(col / ratio)
                out[row, col] = image[i, j]
        return out

    def resize_bilinear_interpolation(self, image, ratio=1.5):
        h, w, c = image.shape
        h_ = int(h*ratio)
        w_ = int(w*ratio)

        out = np.zeros((h_, w_, c), dtype=image.dtype)
        for row in range(h_):
            for col in range(w_):

                i = int(math.floor(row / ratio))
                j = int(math.floor(col / ratio))
                dx = row/ratio - i
                dy = col/ratio - j

                # 防止越界
                if i >=h-1 or j >= w -1:
                    out[row, col] = image[i, j]  
                else:
                    I00 = image[i,j]
                    I10 = image[i+1,j]
                    I01 = image[i, j+1]
                    I11 = image[i+1, j+1]
                    val = (1-dx)*(1-dy)*I00 + dx*(1-dy)*I10 + (1-dx)*dy*I01 + dx*dy*I11

                    out[row, col] = val.astype(np.uint8)
        return out