# import the necessary packages
import numpy as np
import imutils
import cv2
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
from utils import load_image
from transform import *


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()
        self.reinforce_rows = 300
        self.meaningful_threshold = 0.7

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):

        # unpack the images
        (image_left, image_right) = images

        ite_max = 5
        suffice_threshold = 0.1 * image_left.shape[0]

        deg_max = 0.9
        deg_min = 0.6
        for ite in range(ite_max):
            deg = (deg_min + deg_max) / 2

            imageB = np.zeros((image_left.shape[0] + self.reinforce_rows * 2, image_left.shape[1], 4), dtype='uint8')
            imageB[self.reinforce_rows:(self.reinforce_rows + image_left.shape[0]), :, :] = image_left
            imageB = to_diminish_2(imageB, deg) # can optimize

            imageA = np.copy(image_right) # can optimize

            # detect keypoints and extract
            # local invariant descriptors from them
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB,
                                    featuresA, featuresB, ratio, reprojThresh)

            # if the match is None, then there aren't enough matched
            # keypoints to create a Panorama
            if M is None:
                continue

            # otherwise, apply a perspective warp to stitch the images
            # together
            (matches, H, status) = M

            image_A_wrapped = cv2.warpPerspective(imageA, H,
                                         (imageA.shape[1] + imageB.shape[1], imageB.shape[0] + 300),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            (upper_row, lower_row) = self.findBound(image_A_wrapped)
            suffice_index = (lower_row - upper_row) - image_left.shape[0]

            # print "upper_row = {}, lower_row = {}, suffice_index = {}".format(upper_row, lower_row, suffice_index)
            # cv2.imshow('imageB', imageB)
            # cv2.imshow('image_A_wrapped', image_A_wrapped)
            # cv2.waitKey(0)

            if ite == ite_max - 1 or abs(suffice_index) < suffice_threshold:

                print 'abs(suffice_index) = {}'.format(suffice_index)

                # paste the left image to the right one
                result = self.paste(imageB, image_A_wrapped)
                # cv2.imshow('image after pasted', result)
                # cv2.waitKey(0)

                # trim some rows
                result = result[min(upper_row, self.reinforce_rows):max(lower_row, self.reinforce_rows + image_left.shape[0]),:,:]
                # cv2.imshow('image after trimmed Rows', result)
                # cv2.waitKey(0)

                # trim some columns
                result = self.trimSurplusCols(result, minimum_col_index=imageB.shape[1])
                # cv2.imshow('image after trimmed Cols', result)
                # cv2.waitKey(0)

                break

            if suffice_index < suffice_threshold:
                deg_min = deg
            else:
                deg_max = deg

        return result

    def findBound(self, img):
        rows, cols = img.shape[:2]

        zeros = np.zeros(cols, dtype='uint8')

        # print np.sum(img[0, :, 0]), len(img[0, :, 0])
        for i in range(rows):
            if np.array_equal(img[i, :, 0], zeros) == False:
                upper_rows = i
                break

        for i in range(rows - 1, -1, -1):
            if np.array_equal(img[i, :, 0], zeros) == False:
                lower_rows = i
                break
        return (upper_rows, lower_rows)

    def paste(self, left_img, right_img):
        # make the left image fade gradually on the right side
        fade_min = left_img.shape[1] - int(left_img.shape[1] * .2)  # % rightmost of left img will fade gradually
        fade_length = left_img.shape[1] - fade_min

        # print 'fade_min, fade_length = ', fade_min, fade_length
        # print 'left image size:', left_img.shape[:2]
        # print 'right image size:', right_img.shape[:2]

        for j in range(left_img.shape[1]):

            if j >= fade_min:
                a_value = (1. * (left_img.shape[1] - j) / fade_length)
            else:
                a_value = 1
            for i in range (left_img.shape[0]):
                if left_img[i, j, 3] > 0:
                    right_img[i, j, :3] = left_img[i, j, :3] * a_value + (right_img[i, j, :3] * (1 - a_value)).astype(int)
                    right_img[i, j, 3] = 255

        return right_img

    # def trimSurplusRows(self, img, _upper, _lower):
    #     upper = min(_upper, self.reinforce_rows)
    #     lower = max(_lower, self.reinforce_rows + imageB.shape[0])
    #
    #     img = img[upper : (lower + 1), :, :]
    #
    #     return img

    # trim cols must be done after trimmed rows
    def trimSurplusCols(self, img, minimum_col_index):

        for j in range(img.shape[1] - 1, minimum_col_index - 1, -1):
            meaningful = np.sum(img[:, j, 0] != 0) * 1. / img.shape[0]

            if  meaningful > self.meaningful_threshold:
                print 'length:', img.shape[1], ',', 'cut-off to', j, 'because meaningfulness reaches', meaningful, 'percent'
                img = img[:, :j, :]
                break

        return img

    def detectAndDescribe(self, image):
        # print image.shape
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

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
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

    _imageDirectory = '/Users/tungphung/Documents/images5/'
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

    # # stitch CD, BCD, ABCD
    # deg = 0.85
    #
    # imageB = load_image(_images[-1], to_be_diminished_2=False)
    # for i in range (len(_images) - 2, -1, -1):
    #     imageA = load_image(_images[i], to_be_diminished_2=False)
    #     imageA = to_diminish_3(imageA, deg)
    #     # imageB = to_diminish(imageB, 1.6)
    #     cv2.imshow('image after Diminished', imageA)
    #     imageB = stitcher.stitch([imageA, imageB], showMatches=False)
    #
    # # cv2.imwrite('30images.jpg', imageB)
    #     cv2.imshow('Panorama', imageB)
    #     cv2.waitKey()
    #
    #     deg = deg * 0.85

    # stitch AB, CD and ABCD
    deg = 0.85

    imageA = load_image(_images[0])
    imageB = load_image(_images[1])
    imageC = load_image(_images[2])
    imageD = load_image(_images[3])

    # imageA = to_diminish_2(imageA, deg)
    imageAB = stitcher.stitch([imageA, imageB])
    # cv2.imshow('Pano AB', imageAB)

    imageAB = fill_rec(imageAB, imageAB.shape[0])
    cv2.imshow('Pano AB after fill', imageAB)

    imageCD = stitcher.stitch([imageC, imageD])
    # cv2.imshow('Pano CD', imageCD)

    imageCD = fill_rec(imageCD, imageCD.shape[0])

    # cv2.imshow('Pano CD after fill', imageCD)

    # imageAB = to_diminish_2(imageAB, 0.8)
    imageABCD = stitcher.stitch([imageAB, imageCD])
    cv2.imshow('Pano ABCD', imageABCD)

    imageABCD = fill_rec(imageABCD, imageABCD.shape[0])

    cv2.imshow('Pano ABCD after fill', imageABCD)

    cv2.waitKey(0)
