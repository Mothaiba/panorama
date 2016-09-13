# import the necessary packages
from panorama import Stitcher
import argparse
import imutils
import cv2
import os
from os import listdir, mkdir
from os.path import isfile, join, exists

if __name__ == '__main__':
    # # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-f", "--first", required=True,
    #                 help="path to the first image")
    # ap.add_argument("-s", "--second", required=True,
    #                 help="path to the second image")
    # args = vars(ap.parse_args())
    #
    # # load the two images and resize them to have a width of 400 pixels
    # # (for faster processing)
    # imageA = cv2.imread(args["first"])
    # imageB = cv2.imread(args["second"])
    # imageA = imutils.resize(imageA, width=400)
    # imageB = imutils.resize(imageB, width=400)
    #
    # # stitch the images together to create a panorama
    # stitcher = Stitcher()
    # (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    #
    # # show the images
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    # cv2.imshow("Keypoint Matches", vis)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)

    _imageDirectory = '/Users/tungphung/Documents/images8/'
    _imageList = [f for f in listdir(_imageDirectory)]
    _images = [join(_imageDirectory, f) for f in _imageList \
               if isfile(join(_imageDirectory, f)) and not f.startswith('.')]

    if len(_images) < 2:
        print '[>>Error<<] There is %d image' % (len(_images))
        exit(0)
    print '\n\nImage links: ', _images, '\n\n'

    stitcher = Stitcher()

    # imageA = cv2.imread(_images[0])
    #
    # for i in range(1, len(_images)):
    #     imageB = cv2.imread(_images[i])
    #     imageA = imutils.resize(imageA, height=400)
    #     imageB = imutils.resize(imageB, height=400)
    #     (imageA, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    #     # cv2.imshow("Result", imageA)
    #     # cv2.waitKey(0)

    # ===============================================

    #
    # imageB = cv2.imread(_images[-1])
    #
    # for i in reversed(range(0, len(_images) - 1)):
    #     imageA = cv2.imread(_images[i])
    #     imageA = imutils.resize(imageA, height=400)
    #     imageB = imutils.resize(imageB, height=400)
    #     (imageB, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    #     # cv2.imshow("Result", imageB)
    #     # cv2.waitKey(0)
    #
    # print 'imageB size: ', imageB.shape[0], ' ', imageB.shape[1]
    # cv2.imwrite('result.png', imageB)
    #
    # cv2.imshow("Result", imageA)
    # cv2.waitKey(0)

    # ===============================================

    # imageA = imutils.resize(cv2.imread(_images[0]), height = 400)
    # imageB = imutils.resize(cv2.imread(_images[1]), height = 400)
    # imageC = imutils.resize(cv2.imread(_images[2]), height = 400)
    # imageD = imutils.resize(cv2.imread(_images[3]), height = 400)
    #
    # (resultAB, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    # cv2.imshow("ResultAB", resultAB)
    # cv2.waitKey(0)
    #
    # (resultBC, vis) = stitcher.stitch([imageB, imageC], showMatches=True)
    # cv2.imshow("ResultBC", resultBC)
    # cv2.waitKey(0)
    #
    # (resultCD, vis) = stitcher.stitch([imageC, imageD], showMatches=True)
    # cv2.imshow("ResultCD", resultCD)
    # cv2.waitKey(0)
    #
    # (resultABC, vis) = stitcher.stitch([resultAB, resultBC], showMatches=True)
    # cv2.imshow("ResultABC", resultABC)
    # cv2.waitKey(0)
    #
    # (resultBCD, vis) = stitcher.stitch([resultBC, resultCD], showMatches=True)
    # cv2.imshow("ResultBCD", resultBCD)
    # cv2.waitKey(0)
    #
    # (result, vis) = stitcher.stitch([resultABC, resultBCD], showMatches=True)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)

    imageB = imutils.resize(cv2.imread(_images[-1]), height = 400)
    for i in range (len(_images) - 2, -1, -1):
        imageA = imutils.resize(cv2.imread(_images[i]), height=400)

        imageB = stitcher.stitch([imageA, imageB], showMatches=False)

    cv2.imwrite('30images.jpg', imageB)
    cv2.imshow('Something to show', imageB)
    cv2.waitKey()