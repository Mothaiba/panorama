import cv2
import numpy as np


img = cv2.imread('panoramo.jpg', -1)
kernel = np.array([[1, 1, 1],
                   [1, 8, 1],
                   [1, 1, 1]]) / 16.

result = cv2.filter2D(img, -1, kernel)
cv2.imshow('result', result)
cv2.waitKey()