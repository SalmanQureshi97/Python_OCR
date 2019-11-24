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



def SkewCorrection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# Load the image
org_img = cv2.imread("test.jpg")
deskewed_image = SkewCorrection(org_img)
cv2.imwrite('deskewded',deskewed_image)
#PPimg = SkewCorrection(org_img)
#PPimg = NoiseRemoval_Smooth(PPimg)