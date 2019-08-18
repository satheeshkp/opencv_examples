import cv2
import numpy as np
from matplotlib import pyplot as plt
from selectors import BoxSelector

image = cv2.imread("phones.jpg")
bs = BoxSelector(image, "Image")
cv2.imshow("Image", image)
cv2.waitKey(0)

# order the points suitable for the Object detector
pt1, pt2 = bs.roiPts
(x, y, xb, yb) = [pt1[0], pt1[1], pt2[0], pt2[1]]
roi = image[y:yb, x:xb]
# roi = cv2.resize(roi, (60, 90))
cv2.imwrite('template.jpg',roi)