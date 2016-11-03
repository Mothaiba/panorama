# a modification of panorama.py
# improve optimization

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
        self.isv3 = imutils.is_cv3()
        self.reinforce_rows = 100
        self.meaningful_threshold = 0.7
        self.homo = None
        self.ptsA = None
        self.ptsB = None

    def stitch(self, images, ratio=0.6, reprojThresh=4.0, outerTries=30, innerTries = 5):

        def getError(self, deg1_, deg2_):

            self.l_kps = self.ellipse_transform(self.image_left.shape[:2], self.l_ori_kps, deg1_)
            self.r_kps = self.ellipse_transform(self.image_right.shape[:2], self.r_ori_kps, deg2_)
            # print 'Calibration done'

            self.ptsA = np.float32([self.l_kps[i] for (i, _) in self.matches])

            self.ptsB = np.float32([self.r_kps[i] for (_, i) in self.matches])

            # self.drawKps(to_fish_eye(self.image_left, deg_), self.l_kps)

            (H, status) = cv2.findHomography(self.ptsB, self.ptsA, cv2.RANSAC,
                                             reprojThresh)
            # error = abs(H[0, 1]) / 2000. + abs(H[1, 0]) / 500. + abs(H[2, 0]) + abs(H[2, 1]) / 10.

            absH = np.core.abs(np.copy(H))

            error = np.zeros(4)
            # error[0] = 0 if absH[0, 1] < 0.02 else np.power((absH[0, 1] - 0.02) / 2000., 2)
            error[1] = 0 if absH[1, 0] < 0.005 else np.power((absH[1, 0] - 0.005) / 1000, 2)
            error[2] = 0 if absH[2, 0] < 0.00001 else np.power(absH[2, 0] - 0.00001, 2)
            # error[3] = 0 if absH[2, 1] < 0.0001 else np.power((absH[2, 1] - 0.0001) / 10., 2)

            return (sum(error), H)

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

        degs = np.linspace(0, 0.3, outerTries)
        coolest_error = (999, 0, 0, None) # infinite

        for idx, deg1 in enumerate(degs):
            for deg2 in degs[max(0, idx - innerTries) : min(len(degs), idx + innerTries)]:

                (now_error, self.homo) = getError(self, deg1, deg2)
                # print 'deg1: {}, deg2: {}, error: {}'.format(deg1, deg2, now_error)

                if now_error < coolest_error[0]:
                    coolest_error = (now_error, deg1, deg2, self.homo)

        print 'Finally, use deg = {}, {}'.format(coolest_error[1], coolest_error[2])
        deg1 = coolest_error[1]
        deg2 = coolest_error[2]
        self.homo = coolest_error[3]

        if self.homo[0, 2] <= 0:
            return self.image_left if self.image_left.shape[1] > self.image_right.shape[1] else self.image_right


        image_left = to_fish_eye(self.image_left, deg1)
        image_right = to_fish_eye(self.image_right, deg2)

        image_right_wrapped = cv2.warpPerspective(image_right, self.homo,
                                         (image_left.shape[1] + int(self.homo[0, 2]),
                                          image_left.shape[0]),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print 'homo: ', self.homo

        result = self.paste(image_left, image_right_wrapped, moved=self.homo[0, 2])

        return result

    def ellipse_calibration(self, shape, kps, deg):
        (rows, cols) = shape
        a = int(cols / 2)
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
        fade_min = int(moved)  # rightmost of left img will fade gradually
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

    imageA = load_image(_images[0])
    imageB = load_image(_images[1])
    imageC = load_image(_images[2])
    imageD = load_image(_images[3])

    stitcher = Stitcher()
    imageAB = stitcher.stitch([imageA, imageB])
    cv2.imshow('Pano AB', imageAB)
    cv2.imwrite('panoAB.jpg', imageAB)

    stitcher = Stitcher()
    imageCD = stitcher.stitch([imageC, imageD])
    cv2.imshow('Pano CD', imageCD)

    stitcher = Stitcher()
    imageABCD = stitcher.stitch([imageAB, imageCD])
    cv2.imshow('Pano ABCD', imageABCD)

    # cv2.imwrite('panoramo2.jpg', imageABCD)
    cv2.waitKey(0)
