# a modification of panorama6.py
# using to_fish_eye

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
        self.reinforce_rows = 100
        self.meaningful_threshold = 0.7
        self.idenMatrix = np.array([[1, 0], [0, 1]])
        self.homo = None
        self.err_threshold = 0.00001

    def stitch(self, images, ratio=0.6, reprojThresh=4.0, deg=0.5):

        def getError(self, deg_):
            # print 'Get Error'
            # print 'l_kps, r_kps:'
            # print '--len:', len(self.l_kps), len(self.r_kps)
            # print 'matches:', self.matches

            self.l_kps = self.ellipse_transform(self.image_left.shape[:2], self.l_ori_kps, deg_)
            self.r_kps = self.ellipse_transform(self.image_right.shape[:2], self.r_ori_kps, deg_)
            # print 'Calibration done'
            self.ptsA = np.float32([self.l_kps[i] for (i, _) in self.matches])
            self.ptsB = np.float32([self.r_kps[i] for (_, i) in self.matches])

            # self.drawKps(to_fish_eye(self.image_left, deg_), self.l_kps)

            (H, status) = cv2.findHomography(self.ptsB, self.ptsA, cv2.RANSAC,
                                             reprojThresh)
            error = abs(H[1, 0])
            return (error, H)

        # unpack the images
        (self.image_left, self.image_right) = images

        (kps, features) = self.detectAndDescribe(self.image_left)
        self.l_ori_kps = np.copy(kps)
        self.l_kps = np.copy(kps)
        self.l_features = np.copy(features)

        (kps, features) = self.detectAndDescribe(self.image_right)
        self.r_ori_kps = np.copy(kps)
        self.r_kps = np.copy(kps)
        self.r_features = np.copy(features)

        matchResult = self.matchKeypoints(self.r_kps, self.l_kps,
                                          self.r_features, self.l_features, ratio, reprojThresh)

        if matchResult is None:
            print 'images are not match!'
            return None

        (self.matches, H, self.status) = matchResult

        # print 'matches:'
        # print '--len: ', len(self.matches)
        # print '--content: ', self.matches
        # print 'homo'
        # print H
        # cv2.imshow('left', self.image_left)
        # cv2.imshow('warped', cv2.warpPerspective(self.image_right, H, (1000, 1000)))
        # cv2.waitKey(0)
        # vis = self.drawMatches(self.image_right, self.image_left, self.r_kps, self.l_kps, self.matches, self.status)
        # cv2.imshow('vis', vis)
        # cv2.waitKey(0)

        # self.ptsA = np.float32([self.l_kps[i] for (_, i) in self.matches])
        # self.ptsB = np.float32([self.r_kps[i] for (i, _) in self.matches])

        ite_max = 20
        step = deg  # should change to deg
        last_error = 999 # infinite
        coolest_error = (999, 0, None) # infinite

        for ite in range(ite_max):

            print 'deg = {}'.format(deg)

            inc_deg = deg + step
            dec_deg = deg - step

            (now_error, self.homo) = getError(self, deg)
            # print 'homo abc:', self.homo
            # cv2.imshow('warped', cv2.warpPerspective(self.image_right, self.homo, (1000, 1000)))
            # cv2.waitKey(0)
            if now_error < coolest_error[0]:
                coolest_error = (now_error, deg, self.homo)

            (inc_error, inc_homo) = getError(self, inc_deg)
            (dec_error, dec_homo) = getError(self, dec_deg)

            if inc_error < dec_error:
                deg = deg + step / 2.
            else:
                deg = deg - step / 2.

            step = step / 2.

            # print 'current error = {}, dec error = {}, inc error = {}'.format(now_error, dec_error, inc_error)

            if abs(now_error - last_error) < self.err_threshold:
                # print 'now_error = {}, break after {} iterations'.format(now_error, ite + 1)
                break

            last_error = now_error

        print 'Finally, use deg = {}'.format(coolest_error[1])
        deg = coolest_error[1]
        self.homo = coolest_error[2]
        # print self.homo
        if self.homo[0, 2] <= 0:
            return self.image_left if self.image_left.shape[1] > self.image_right.shape[1] else self.image_right


        image_left = to_fish_eye(self.image_left, deg)
        image_right = to_fish_eye(self.image_right, deg)

        image_right_wrapped = cv2.warpPerspective(image_right, self.homo,
                                         (image_left.shape[1] + int(self.homo[0, 2]),
                                          image_left.shape[0]),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print 'homo: ', self.homo
        cv2.imshow('left', image_left)
        cv2.imshow('right', image_right_wrapped)

        result = self.paste(image_left, image_right_wrapped, moved=self.homo[0, 2])

        # cv2.imshow('result', result)
        # cv2.waitKey(0)

        return result

    def calibrate(self, shape, kps, deg):
        (rows, cols) = shape
        # print 'rows, cols:', rows, cols
        a = cols / 2
        b = a * deg

        centre = cols / 2.
        diff_centre = lambda g: g - centre if g > centre else centre - g

        new_kps = []
        for i in range(len(kps)):
            y, x = kps[i]
            add_height = int(((1. - (1. * diff_centre(y) / a) ** 2) * (b ** 2)) ** 0.5)
            height_ratio = 1 + (add_height * 2. / rows)

            min_row_in_consider = b - add_height
            new_x = min_row_in_consider + int(x * height_ratio)

            new_kps.append([y, new_x])

        # print 'a, b, deg:', a, b, deg
        # print kps
        # print new_kps
        # cv2.waitKey(0)
        return new_kps

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


    def paste(self, left_img, right_img, moved):
        fade_min = moved  # rightmost of left img will fade gradually
        fade_length = left_img.shape[1] - fade_min

        for j in range(left_img.shape[1]):

            if j >= fade_min:
                a_value = (1. * (left_img.shape[1] - j) / fade_length)
            else:
                a_value = 1
            for i in range (left_img.shape[0]):
                if left_img[i, j, 3] > 0:
                    right_img[i, j, :3] = left_img[i, j, :3] * a_value + (right_img[i, j, :3] * (1 - a_value)).astype(int)
                    right_img[i, j, 3] = 255

        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 2, 2, 2, 1],
                           [1, 2, 5, 2, 1],
                           [1, 2, 2, 2, 1],
                           [1, 1, 1, 1, 1]]) / 37.

        right_img[:, (fade_min):(fade_min + 30), :3] = cv2.filter2D(src=right_img[:, (fade_min):(fade_min + 30), :3],
                                                                    ddepth=-1, kernel=kernel)
        # right_img[:, (fade_min - 10):(fade_min + 10), :3] = 255


        return right_img

    def paste_2(self, left_img, right_img, moved):

        centre_left = left_img.shape[1] / 2
        centre_right = (right_img.shape[1] + moved) / 2
        real_cols_right = (right_img.shape[1] - moved)
        real_cols_left = left_img.shape[1]

        rows, cols = right_img.shape[:2]

        getDiffLeff = lambda g: 1. * (g - centre_left) / real_cols_left if g > centre_left else 1. * (centre_left - g) / real_cols_left
        getDiffRight = lambda g: 1. * (g - centre_right) / real_cols_right if g > centre_right else 1. * (centre_right - g) / real_cols_right

        for i in range(rows):
            for j in range(cols):
                if j >= real_cols_left or left_img[i, j, 0] == 0:
                    pass
                elif right_img[i, j, 0] == 0:
                    right_img[i, j, :] = left_img[i, j, :]

                else:
                    prop_l = 1. - getDiffLeff(j)
                    prop_r = 1. - getDiffRight(j)
                    sum_diff = prop_l + prop_r
                    right_img[i, j, :] = (left_img[i, j, :] * prop_l / sum_diff + right_img[i, j, :] * prop_r / sum_diff).astype(int)

        # right_img[:, :, 3] = 255
        return right_img

    def paste_3(self, left_img, right_img, moved):
        fade_min = int(moved)  # rightmost of left img will fade gradually
        fade_length = left_img.shape[1] - fade_min

        right_img[:, :fade_min, :] = left_img[:, :fade_min, :]

        for j in range(fade_min, left_img.shape[1]):
            for i in range(left_img.shape[0]):
                if right_img[i, j, 0] != 0 and left_img[i, j, 0] != 0:
                    right_img[i, j, :3] = (left_img[i, j, :3] + right_img[i, j, 3]) / 2
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
        vis = np.zeros((max(hA, hB), wA + wB, 4), dtype="uint8")
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

    def drawKps(self, image, kps):
        # print 'Drawing'
        for kp in kps:
            cv2.circle(image, (int(kp[0]), int(kp[1])), 5, (0, 255, 0))
        cv2.imshow('kps', image)
        cv2.waitKey(0)


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

    _imageDirectory = '/Users/tungphung/Documents/images15/'
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

    imageA = load_image(_images[0])
    imageB = load_image(_images[1])
    imageC = load_image(_images[2])
    imageD = load_image(_images[3])

    # imageA = to_diminish_2(imageA, deg)
    imageAB = stitcher.stitch([imageA, imageB], deg=0.5)
    cv2.imshow('Pano AB', imageAB)

    # imageAB = fill_rec(imageAB, imageAB.shape[0])
    # cv2.imshow('Pano AB after fill', imageAB)

    imageCD = stitcher.stitch([imageC, imageD], deg=0.3)
    cv2.imshow('Pano CD', imageCD)
    # cv2.waitKey(0)
    # imageCD = fill_rec(imageCD, imageCD.shape[0])

    # cv2.imshow('Pano CD after fill', imageCD)

    # cv2.waitKey(0)

    # imageAB = to_diminish_2(imageAB, 0.8)
    imageABCD = stitcher.stitch([imageAB, imageCD], deg=0.3)
    cv2.imshow('Pano ABCD', imageABCD)

    # imageABCD = fill_rec(imageABCD, imageABCD.shape[0])

    # cv2.imshow('Pano ABCD after fill', imageABCD)
    cv2.imwrite('panoramo2.jpg', imageABCD)
    cv2.waitKey(0)
