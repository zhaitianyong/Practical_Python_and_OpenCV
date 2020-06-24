import cv2
import numpy as np


samples_data_path = "/home/atway/soft/opencv4.1/opencv-4.1.0/samples/data/"

if __name__== "__main__":
    print(cv2.__version__)
    image = cv2.imread(samples_data_path + "lena.jpg")
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()