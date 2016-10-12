# a modification of panorama6.py
# use Global Optimization
# boost speed
# NOT completed

# import the necessary packages
import numpy as np
import imutils
import cv2
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
from utils import load_image
from transform import *

INFINITY = 999999999

# NOTE: each Stitcher object can only stitch 1 time
class Stitcher:
    def __init__(self):

        self.perfectHomo = np.array([[1, 0], [0, 1], [0, 0]])
        self.original_keypoints = []
        self.keypoints = []
        self.features = []
        self.images = None
        self.matches = {}
        self.homo = {}
        self.status = {}
        self.nimages = None
        self.ellipseDeg = None
        self.loops = 5
        self.defaultDeg = 0.4

    def stitch(self, images, ratio=0.4, reprojThresh=4.0):

        # save the images
        self.images = images

        # temporary only process cases where the number of images is 4, and all have the same shape
        if len(self.images) != 4:
            return None
        for i in range(len(self.images)):
            if not np.array_equal(np.array(self.images[0].shape), np.array(self.images[i].shape)):
                return None

        # initialize some values
        self.nimages = len(self.images)
        self.ellipseDeg = [self.defaultDeg] * self.nimages
        self.incDeg = self.ellipseDeg * 0.1
        self.decDeg = self.ellipseDeg * 0.1

        # find the key-points of each image
        for image in self.images:
            (kps, features) = self.detectAndDescribe(image)
            self.original_keypoints.append(kps)
            self.keypoints.append(kps)
            self.features.append(features)

            # self.drawKeypoints(image, kps)

        # initialize calibration on images
        for i in range(self.nimages):
            self.calibrate(idx=i, updateLeft=False, updateRight=False)

        # construct self.homo, self.matches, self.status
        for i in range(len(self.images) - 1):

            matchResult = self.matchKeypoints(self.original_keypoints[i], self.original_keypoints[i + 1],
                                self.features[i], self.features[i + 1], ratio, reprojThresh)

            #  if no matching between a pair of consecutive images, return to imply faulty
            if matchResult is None:
                print 'matchResult == None'
                return None

            (matches, H, status) = matchResult
            # print 'matches length:', len(matches)

            self.matches[str(i) + '-' + str(i + 1)] = matches
            self.matches[str(i + 1) + '-' + str(i)] = [(p2, p1) for (p1, p2) in matches]

            self.homo[str(i) + '-' + str(i + 1)] = np.mat(np.linalg.inv(H))
            self.homo[str(i + 1) + '-' + str(i)] = np.mat(H)

            self.status[str(i) + '-' + str(i + 1)] = status
            self.status[str(i + 1) + '-' + str(i)] = status

            # print 'mat ivH: ', self.homo[str(i) + '-' + str(i + 1)]
            # print self.homo[str(i) + '-' + str(i + 1)] * self.homo[str(i + 1) + '-' + str(i)]
            # print self.matches[str(i) + '-' + str(i + 1)]
            # print self.matches[str(i + 1) + '-' + str(i)]

            # vis = self.drawMatches(self.images[i], self.images[i + 1], self.keypoints[i],
            #                  self.keypoints[i + 1], self.matches[str(i) + '-' + str(i + 1)], self.status[str(i) + '-' + str(i + 1)])
            # cv2.imshow('matches', vis)
            # cv2.waitKey(0)

        # for gg in range(4):
        #     images[gg] = to_fish_eye(self.images[gg], self.ellipseDeg[gg])
        # cv2.imshow('img0', images[0])
        # cv2.imshow('img1', cv2.warpPerspective(images[1], self.homo['0-1'], (600, 700)))
        # cv2.imshow('img2', cv2.warpPerspective(images[2], self.homo['0-1'] * self.homo['1-2'], (1200, 700)))
        # cv2.waitKey(0)
        # exit(0)

        # do global optimization
        for loop in range(self.loops):
            idx = np.random.random_integers(0, self.nimages - 1)
            self.tryModify(idx=idx, keypoints=self.keypoints, homo=self.homo, currentDeg=self.ellipseDeg,
                           incDeg=self.incDeg, decDeg=self.decDeg, toLeft=True, toRight=True)

    def tryModify(self, idx, keypoints, homo, currentDeg, incDeg, decDeg, toLeft, toRight):
        # return error = 0 if out of range
        if idx < 0 or idx >= self.nimages:
            return None

        local_min_error = INFINITY

        inc_currentDeg = np.copy(currentDeg)
        inc_currentDeg[idx] = currentDeg[idx] + incDeg[idx]
        inc_keypoints = self.changeKpsByDeg(idx=idx, deg=inc_currentDeg[idx])

        dec_currentDeg = np.copy(currentDeg)
        dec_currentDeg[idx] = currentDeg[idx] + decDeg[idx]
        dec_keypoints = self.changeKpsByDeg(idx=idx, deg=dec_currentDeg[idx])

        if (toLeft and toRight):
            inc_error = self.tryModify(idx=idx-1, keypoints=inc_keypoints, homo=homo, currentDeg=inc_currentDeg,
                                       incDeg=incDeg, decDeg=decDeg, toLeft=True, toRight=False)\
                        +\
                        self.tryModify(idx=idx+1, keypoints=inc_keypoints, homo=homo, currentDeg=inc_currentDeg,
                                       incDeg=incDeg, decDeg=decDeg, toLeft=False, toRight=True)

            dec_error = self.tryModify(idx=idx-1, keypoints=dec_keypoints, homo=homo, currentDeg=dec_currentDeg,
                                       incDeg=incDeg, decDeg=decDeg, toLeft=True, toRight=False)\
                        +\
                        self.tryModify(idx=idx+1, keypoints=dec_keypoints, homo=homo, currentDeg=dec_currentDeg,
                                       incDeg=incDeg, decDeg=decDeg, toLeft=False, toRight=True)
            if inc_error < dec_error:
                if inc_error < local_min_error:
                    ?? luu lai error, homo, ellipseDeg

                currentDeg = inc_currentDeg
                keypoints = inc_keypoints


            else:


        elif toLeft:
            pass
        elif toRight:
            pass

        return local_min_error

    def mergeResult(self, resultLeft, resultMid, resultRight, idx):
        if resultLeft == None:
            return resultRight


    def changeKpsByDeg(self, idx, keypoints, deg):
        kps_full = np.copy(keypoints)
        kps = np.copy(self.original_keypoints[idx])

        rows, cols = self.images[idx].shape[:2]
        # print 'rows, cols:', rows, cols
        a = cols / 2
        b = a * self.ellipseDeg[idx]

        centre = cols / 2.
        diff_centre = lambda g: g - centre if g > centre else centre - g

        for i in range(len(self.original_keypoints[idx])):
            y, x = self.original_keypoints[idx][i]
            add_height = int(((1. - (1. * diff_centre(y) / a) ** 2) * (b ** 2)) ** 0.5)
            height_ratio = 1 + (add_height * 2. / rows)

            min_row_in_consider = b - add_height
            new_x = min_row_in_consider + int(x * height_ratio)

            kps[i] = [y, new_x]
            # print x, new_x, y

        kps_full[idx] = np.copy(kps)
        return kps_full


    def calibrate(self, idx):
        rows, cols = self.images[idx].shape[:2]
        # print 'rows, cols:', rows, cols
        a = cols / 2
        b = a * self.ellipseDeg[idx]

        centre = cols / 2.
        diff_centre = lambda g: g - centre if g > centre else centre - g

        for i in range(len(self.original_keypoints[idx])):
            y, x = self.original_keypoints[idx][i]
            add_height = int(((1. - (1. * diff_centre(y) / a) ** 2) * (b ** 2)) ** 0.5)
            height_ratio = 1 + (add_height * 2. / rows)

            min_row_in_consider = b - add_height
            new_x = min_row_in_consider + int(x * height_ratio)

            self.keypoints[idx][i] = [y, new_x]
            # print x, new_x, y


    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # print 'kps:', kps
        # print 'features:', features

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

    def drawKeypoints(self, img, kps):
        for kp in kps:
            cv2.circle(img, (kp[0], kp[1]), 5, (0, 255, 0))

        cv2.imshow('keypoints', img)
        cv2.waitKey(0)

if __name__ == '__main__':

    _imageDirectory = '/Users/tungphung/Documents/images5/'
    _imageList = [f for f in listdir(_imageDirectory)]
    _images = [join(_imageDirectory, f) for f in _imageList \
               if isfile(join(_imageDirectory, f)) and not f.startswith('.')]

    if len(_images) < 2:
        print '[>>Error<<] There is %d image' % (len(_images))
        exit(0)
    print '\n\nImage links: ', _images, '\n\n'

    images = []
    for _image in _images:
        images.append(load_image(_image))

    ratio = 0.3
    while True:
        stitcher = Stitcher()
        pano = stitcher.stitch(images, ratio=ratio)

        if pano is not None:
            break
        ratio = ratio + 0.1
        print 'use Lowe\'s ration =', ratio

    cv2.imshow('pano', pano)
    cv2.waitKey(0)