import cv2
import numpy as np
import imutils
import cv2
from os import listdir, mkdir
from os.path import isfile, join, exists


_imageDirectory = '/Users/tungphung/Documents/images3/'
_imageList = [f for f in listdir(_imageDirectory)]
_images = [join(_imageDirectory, f) for f in _imageList \
           if isfile(join(_imageDirectory, f)) and not f.startswith('.')]

images = []

for _image in _images:
    images.append(imutils.resize(cv2.imread(_image, -1), height=400))
    # cv2.imshow('Image', images[-1])
    # print images[-1]
    print _image

print 'len(images) =', len(images)

stitcher1 = cv2.createStitcher(False)

# pairwise = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
# stitcher1.setMatchingMask(pairwise)

result1 = stitcher1.stitch(images[:4])
print 'status code: ', result1[0]
cv2.imshow('Pano1', result1[1])
cv2.waitKey(0)



# stitcher2 = cv2.createStitcher(False)
# result2 = stitcher2.stitch(images)
# cv2.imshow('Pano2', result2[1])
# cv2.waitKey(0)

# cv2.imwrite('/Users/tungphung/panor.png', result[1])