from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
val = 0


def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.GaussianBlur(src_gray, (17, 17), 5)
    img_dilated = cv.dilate(img_blur, kernel=np.ones((5, 5), np.uint8), iterations=2)
    retval, img_bw = cv.threshold(img_blur, 7, 255, cv.THRESH_BINARY)
    detected_edges = cv.Canny(img_dilated, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)


parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
val = cv.getTrackbarPos(title_trackbar, window_name)
print(val)
cv.waitKey()
val = cv.getTrackbarPos(title_trackbar, window_name)
print(val)