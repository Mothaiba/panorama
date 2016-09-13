import sys

import cv2
import numpy as np
import imutils
import cv2
from os import listdir, mkdir
from os.path import isfile, join, exists

def sidest_matches(img1, keypoints1, img2, keypoints2, matches, n_nearest, leftmost = False):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    new_matches = []

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        pos = (cols1 - x1) + (cols2 - x2)

        new_matches.append({
            'queryIdx' : img1_idx,
            'trainIdx' : img2_idx,
            'pos' : pos
        })


    new_matches = sorted(new_matches, key=lambda x: x['pos'], reverse=leftmost)[:n_nearest]


    return new_matches

def concatenate(output_img, img1, rows1, cols1, img2, rows2, cols2, x_diff, y_diff):

    fade_length = int(cols2 * 0.0) # 1% leftmost of img2 will fade gradually

    for j in range(fade_length):
        trans_deg = 1. * j / fade_length
        for i in range(rows2):
            img2[i, j, :] = np.array((img2[i, j, :] * trans_deg + img1[i - x_diff, j - y_diff, :] * (1 - trans_deg))).astype(int)


    if x_diff > 0:
        output_img[x_diff : (x_diff + rows1), 0 : cols1, :] = img1
        output_img[0 : rows2, (-y_diff) : (-y_diff + cols2), :] = img2
    else:
        output_img[0 : rows1, 0 : cols1, :] = img1
        output_img[(-x_diff) : (-x_diff + rows2), (-y_diff) : (-y_diff + cols2), :] = img2

    return output_img

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Create a new output image that concatenates the two images together
    output_img = np.zeros((max([rows1,rows2]), cols1+cols2, 4), dtype='uint8')
    output_img[:rows1, :cols1, :] = img1
    output_img[:rows2, cols1:cols1+cols2, :] = img2

    # Draw connecting lines between matching keypoints
    for match in matches:
        # Get the matching keypoints for each of the images
        img1_idx = match['queryIdx']
        img2_idx = match['trainIdx']

        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw a small circle at both co-ordinates and then draw a line
        radius = 4
        colour = (0,255,0)   # green
        thickness = 1
        cv2.circle(output_img, (int(x1),int(y1)), radius, colour, thickness)
        cv2.circle(output_img, (int(x2)+cols1,int(y2)), radius, colour, thickness)
        cv2.line(output_img, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), colour, thickness)

    return output_img

def stitch_images(img1, keypoints1, img2, keypoints2, matches):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    x_diff = 0
    y_diff = 0

    # Draw connecting lines between matching keypoints
    for match in matches:
        # Get the matching keypoints for each of the images
        img1_idx = match['queryIdx']
        img2_idx = match['trainIdx']

        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        x_diff = x_diff + (y2 - y1)
        y_diff = y_diff + (x2 - x1)


    x_diff = int(x_diff / len(matches))
    y_diff = int(y_diff / len(matches))

    print 'x_diff, y_diff = ', x_diff, ', ', y_diff

    if x_diff > 0:
        nrow = rows1 + x_diff
    else:
        nrow = max(rows1, rows2 - x_diff)
    ncol = cols2 - y_diff

    print 'stitched image: (nrows, ncols) = ' + '(' + str(nrow) + ', ' + str(ncol) + ')'

    output_img = np.zeros((nrow, ncol, 4), dtype='uint8')

    return concatenate(output_img, img1, rows1, cols1, img2, rows2, cols2, x_diff, y_diff)



if __name__=='__main__':

    # img1 = imutils.resize(cv2.imread('/Users/tungphung/Documents/images7/P_20160913_100320.jpg'), height=400)
    # img2 = imutils.resize(cv2.imread('/Users/tungphung/Documents/images7/P_20160913_100323.jpg'), height=400)
    # # Initialize ORB detector
    # orb = cv2.ORB_create()
    #
    # # Extract keypoints and descriptors
    # keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    #
    # # Create Brute Force matcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #
    # # Match descriptors
    # matches = bf.match(descriptors1, descriptors2)
    #
    # # Sort them in the order of their distance
    # matches = sorted(matches, key = lambda x:x.distance)
    #
    # # Draw first 'n' matches
    # print 'There are ' + str(len(matches)) + ' matched points'
    # img3 = draw_matches(img1, keypoints1, img2, keypoints2, matches[:30])
    # img4 = stitch_images(img1, keypoints1, img2, keypoints2, matches[:30])
    #
    # cv2.imshow('Matched keypoints', img3)
    # cv2.imshow('Panorama', img4)
    # cv2.waitKey()


    _imageDirectory = '/Users/tungphung/Documents/images7/'
    _imageList = [f for f in listdir(_imageDirectory)]
    _images = [join(_imageDirectory, f) for f in _imageList \
               if isfile(join(_imageDirectory, f)) and not f.startswith('.')]

    if len(_images) < 2:
        print '[>>Error<<] There is %d image' % (len(_images))
        exit(0)
    print '\n\nImage links: ', _images, '\n\n'

    # Initialize ORB detector
    orb = cv2.ORB_create()

    img = imutils.resize(cv2.imread(_images[0], -1), height=400)
    img1 = np.full((img.shape[0], img.shape[1], 4), 255, dtype='uint8')
    img1[:,:,:3] = img

    for i in range(1, len(_images)):
        img = imutils.resize(cv2.imread(_images[i], -1), height=400)
        img2 = np.full((img.shape[0], img.shape[1], 4), 255, dtype='uint8')
        img2[:, :, :3] = img

        # Extract keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # Create Brute Force matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort them in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)

        # From the most similar matches, take the rightmost matches
        matches = sidest_matches(img1, keypoints1, img2, keypoints2, matches[:99], 30, leftmost=True)

        # Draw first 'n' matches
        # img3 = draw_matches(img1, keypoints1, img2, keypoints2, matches)
        # cv2.imshow('Matched keypoints', img3)
        # cv2.waitKey()
        img1 = stitch_images(img1, keypoints1, img2, keypoints2, matches)

    # cv2.imshow('Matched keypoints', img3)
    cv2.imshow('Panorama', img1)
    cv2.waitKey()