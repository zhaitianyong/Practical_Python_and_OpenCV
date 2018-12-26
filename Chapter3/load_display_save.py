import cv2
from __future__ import print_function
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image", default="/home/ml/data/lena.jpg")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey()



