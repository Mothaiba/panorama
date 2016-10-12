import cv2
import numpy as np
import imutils

raw = imutils.resize(cv2.imread('/Users/tungphung/Documents/images15/P_20160918_093036.jpg', -1), height=400)
r, c = raw.shape[:2]

img = np.zeros((1200, 1200, 3), dtype='uint8')
img[200:200 + r, 200:200 + c, :] = raw

homo = np.array([[1, 0., 0],
                 [-0., 1, 0],
                 [-0.0009, 0.0000, 1]], dtype=float)

homo_m = np.matrix(homo)
print 'homo invert:', np.linalg.inv(homo_m)

wrapped = cv2.warpPerspective(img, homo, (1200, 1000), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
cv2.imshow('image', wrapped)
cv2.waitKey()