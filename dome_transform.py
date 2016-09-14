import numpy as np
import cv2
import imutils

def to_dome(img, _b):
    rows, cols = img.shape[:2]
    a = cols / 2
    b = int(_b * a)

    print 'rows, cols = ' + str(rows), ', ', str(cols)
    print 'a, b = ' + str(a), ', ', str(b)

    dome_rows = rows + b
    dome = np.full((dome_rows, cols, 4), 255, dtype='uint8')
    diff_centre = lambda x : x - a if x > a else a - x

    for j in range (cols):
        add_height = int(((1. - (1. * diff_centre(j) / a) ** 2) * (b ** 2)) ** 0.5)
        # print dome_rows - add_height - rows, rows
        dome[(dome_rows - add_height - rows) : (dome_rows - add_height), j, :3] = img[:, j, :3]

    return dome

def to_dome_2(img, _major, _minor):
    rows, cols = img.shape[:2]
    a = cols / 2
    major = int(_major * a)
    minor = int(_minor * a)

    print 'rows, cols = ' + str(rows), ', ', str(cols)
    print 'a, major, minor = ' + str(a), ', ', str(major), ', ', str(minor)

    dome_rows = rows + major
    dome = np.full((dome_rows, cols, 4), 255, dtype='uint8')
    diff_centre = lambda x: x - a if x > a else a - x

    for j in range(cols):
        diff = (1. - (1. * diff_centre(j) / a) ** 2)
        # print 'diff = ', diff
        major_add = int((diff * (major ** 2)) ** 0.5)
        minor_add = int((diff * (minor ** 2)) ** 0.5)

        # print 'major_add, minor_add = ', major_add, minor_add

        ratio = 1 + 1. * (major_add - minor_add) / rows

        # print 'ratio: ', str(ratio)

        min_row_in_consider = dome_rows - major_add - rows
        for i in range(dome_rows - major_add - rows, dome_rows - minor_add):
            dome[i, j, :3] = img[int((i - min_row_in_consider) / ratio), j, :]
    return dome

if __name__ == '__main__':
    # tmp_img = imutils.resize(cv2.imread('Panorama_lab_3_pics.png', -1), height=400)
    # img = np.full((tmp_img.shape[0], tmp_img.shape[1], 4), 255, dtype='uint8')
    # img[:, :, :3] = tmp_img
    img = imutils.resize(cv2.imread('/Users/tungphung/Documents/images8/P_20160913_100320.jpg', -1), height=400)

    # dome = to_dome(img, 1.5)
    # cv2.imshow('Dome', dome)

    dome_2 = to_dome_2(img, 0.8, 0.1)
    cv2.imshow('Dome 2', dome_2)

    cv2.waitKey(0)