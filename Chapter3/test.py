import cv2
import numpy as np



if __name__== "__main__":
    print(cv2.__version__)
    image = cv2.imread("/home/ml/data/lena.jpg")
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()