import cv2
import numpy as np
import imutils
import math
import datetime

def spherical_calibration(img, f):
    rows, cols = img.shape[:2]
    print 'rows, cols:', rows, cols

    x_mid = int(cols / 2)
    y_mid = int(rows / 2)
    s = f
    result = np.zeros((rows, cols, 3), dtype='uint8')

    for y in range(rows):
        y_mirror = y - y_mid
        for x in range(cols):
            x_mirror = x - x_mid
            x_ = int(s * math.atan2(x_mirror, f))
            y_ = int(s * math.atan2(y_mirror, math.sqrt(x_mirror * x_mirror + f * f)))
            result[y_ + y_mid, x_ + x_mid] = img[y, x]

    cv2.imshow('result time = {}'.format(datetime.datetime.now()), result)



if __name__ == '__main__':
    f = 400
    img1 = imutils.resize(cv2.imread('/Users/tungphung/Documents/images15/P_20160918_093036.jpg'), height=f)
    img2 = imutils.resize(cv2.imread('/Users/tungphung/Documents/images15/P_20160918_093047.jpg'), height=f)
    spherical_calibration(img1, f)
    spherical_calibration(img2, f)

    cv2.waitKey(0)