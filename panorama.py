# import the necessary packages
import numpy as np
import imutils
import cv2
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
from utils import load_image


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a Panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        # result = cv2.warpPerspective(imageA, H,
        #                              (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + 400, int(imageA.shape[0] + 200)))

        # result = cv2.warpPerspective(imageA, H,
        #                              (imageA.shape[1] + 20, int(imageA.shape[0] + 15)))

        print 'imageA size: ', imageA.shape[0], ' ', imageA.shape[1]
        print 'imageB size: ', imageB.shape[0], ' ', imageB.shape[1]
        print '\n'

        cv2.imshow('raw merge', result)

        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis


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

    imageB = load_image(_images[-1], to_be_diminished=True)
    for i in range (len(_images) - 2, -1, -1):
        imageA = load_image(_images[i], to_be_diminished=True)
        imageB = stitcher.stitch([imageA, imageB], showMatches=False)

    # cv2.imwrite('30images.jpg', imageB)
    cv2.imshow('Something to show', imageB)
    cv2.waitKey()