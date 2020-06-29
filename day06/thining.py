import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from  day06.morphology import *


def Thin(img):
    pass





if __name__=="__main__":
    image = cv2.imread("../data/day05/imori.jpg")
    gray = BGR2GRAY(image)

    binary = OSTU(gray)
    out = Gradient(binary)

    cv2.imshow("out", out)
    cv2.waitKey()