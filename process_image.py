#!/usr/bin/python

import cv2
import numpy as np
import imutils as imutils

def NoiseRemoval_Smooth(img):
   
    filtered = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = cv2.GaussianBlur(filtered,(3,3),0)
    or_image = cv2.bitwise_or(img, closing)
    cv2.imwrite('Denoised.jpg',or_image)
    return or_image 


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
def SkewCorrection(Image):
   gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
   edged = cv2.Canny(gray, 10, 50)
   cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts = cnts[0] if imutils.is_cv2() else cnts[1]
   cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
   screenCnt = None
   for c in cnts:
       peri = cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, 0.02 * peri, True)
       if len(approx) == 4:
           screenCnt = approx
           break
    
   cv2.drawContours(Image, [screenCnt], -1, (0, 255, 0), 2)
#   pts = np.array(screenCnt.reshape(4, 2) * ratio)
 #  warped = four_point_transform(orig, pts)

# Load the image
org_img = cv2.imread("test.jpg",cv2.IMREAD_GRAYSCALE)
deskewed_image = SkewCorrection(org_img.copy())
cv2.imwrite('deskewded',deskewed_image)
#PPimg = SkewCorrection(org_img)
#PPimg = NoiseRemoval_Smooth(PPimg)