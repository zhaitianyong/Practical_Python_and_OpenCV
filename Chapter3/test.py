import cv2
import numpy as np



if __name__== "__main__":
    image = cv2.imread("/home/ml/data/lena.jpg")
    cv2.imshow("image", image)
    cv2.waitKey()