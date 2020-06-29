import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class TemplateMatch(object):
    def __init__(self):
        print("template match")

    def SSD(self, img, template):
        H, W, C = img.shape
        h, w, c = template.shape
        match_img = np.zeros((H-h, W-w), np.float)
        for i in range(H-h):
            for j in range(W-w):
                block = img[i:i+h, j:j+w, :]
                diff = block - template
                diff_2 = diff * diff
                match_img[i, j] = np.sum(diff_2)
        minValue = np.min(match_img)
        indx = np.where(match_img==minValue)
        print(indx)
        return (indx[0], indx[1])


    def SAD(self, img, template):
        H, W, C = img.shape
        h, w, c = template.shape
        match_img = np.zeros((H - h, W - w), np.float)
        for i in range(H - h):
            for j in range(W - w):
                block = img[i:i+h, j:j+w, :]
                diff = np.abs(block - template)
                match_img[i, j] = np.sum(diff)
        minValue = np.min(match_img)
        indx = np.where(match_img == minValue)
        print(indx)

        return (indx[0], indx[1])

    def NCC(self, img, template):
        H, W, C = img.shape
        h, w, c = template.shape
        match_img = np.zeros((H - h, W - w), np.float)
        for i in range(H - h):
            for j in range(W - w):
                block = img[i:i + h, j:j + w, :]
                sqrt_b = np.sqrt(np.sum(block*block))
                sqrt_t = np.sqrt(np.sum(template*template))
                s = (np.sum(block*template)/ sqrt_b) / sqrt_t
                match_img[i, j] = s
        maxValue = np.max(match_img)
        indx = np.where(match_img == maxValue)
        print(indx)

        return (indx[0], indx[1])

    def ZNCC(self, img, template):
        H, W, C = img.shape
        h, w, c = template.shape
        match_img = np.zeros((H - h, W - w), np.float)
        u0 = np.mean(template)
        for i in range(H - h):
            for j in range(W - w):
                block = img[i:i + h, j:j + w, :]
                u1 = np.mean(block)
                sqrt_b = np.sqrt(np.sum(np.power(block-u1, 2)))
                sqrt_t = np.sqrt(np.sum(np.power(template-u0, 2)))
                s = (np.sum((block-u1) * (template-u0)) / sqrt_b) / sqrt_t
                match_img[i, j] = s
        maxValue = np.max(match_img)
        indx = np.where(match_img == maxValue)
        print(indx)

        return (indx[0], indx[1])


if __name__ == "__main__":
    image = cv2.imread("../data/day06/imori.jpg")
    template = cv2.imread("../data/day06/imori_part.jpg")
    h, w = template.shape[0], template.shape[1]
    templateMatch = TemplateMatch()
    index  = templateMatch.SSD(image, template)

    #  (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255),
    cv2.rectangle(image, index, (index[0]+h, index[1]+w), (0, 0, 255), 2)
    cv2.imshow("out", image)
    cv2.waitKey()
