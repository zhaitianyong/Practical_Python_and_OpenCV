
import os
import cv2
import numpy as np
from day02.Filter import Filter



sampel_data_path = os.path.join("../data/day01/")



if __name__ == "__main__":
    filter = Filter()
    filter.test()
    image = cv2.imread(sampel_data_path + "lena.jpg", cv2.IMREAD_GRAYSCALE)
    # out = filter.padding(image, 7)
    # cv2.imshow("out", out)
    # cv2.waitKey()
    #out = filter.meanFilter(image, 7)
    #out = filter.motionFilter(image, 7)
    #out = filter.minMaxFilter(image, 5)
    # out1 = filter.prewittFilter(image, axis=0)
    # out2 = filter.prewittFilter(image, axis=1)
    # result1 = np.abs(out1).astype(np.uint8)
    # result2 = np.abs(out2).astype(np.uint8)
    #
    # merge = (result1*0.5+ result2*0.5).astype(np.uint8)
    # result = np.hstack((result1, result2,merge))
    # result= filter.embossFilter(image)
    # cv2.imshow("out", result)
    # cv2.waitKey()
    filter.logFilter(image)