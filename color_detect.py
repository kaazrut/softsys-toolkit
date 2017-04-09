#import packages
import numpy as np
import argparse
import cv2
import os
from operator import itemgetter
from imutils import perspective
from imutils import contours as cnts
import ipdb

def cdetect(basepath, currentimage, testFlag):

    path = os.path.join(basepath, currentimage)
    image = cv2.imread(path)

    #compression
    if testFlag == 1: 
        #define boundaries of RGB (in BGR order)
        boundaries = ([17,15,95], [90, 100, 255]) #red

        lower = np.array(boundaries[0], dtype="uint8")
        upper = np.array(boundaries[1], dtype="uint8")
        #find colors within boundaries, apply mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        ret, thresh1 = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(thresh1, kernel, iterations = 1)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        #determine pixel coordinates
        _, bounds, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(closing, bounds, -1, (255, 255, 255), 3)
        com = []
        threshpass = []
        threshold_area = 1000
        for i, cnt in enumerate(bounds):
            area = cv2.contourArea(cnt)
            if area > threshold_area:
                threshpass.append((area, bounds[i]))

        threshpass.sort(key=itemgetter(0), reverse=True)
        for x in range(0, 2):
            moi = cv2.moments(threshpass[x][1])
            com.append((int(moi['m10']/moi['m00']), int(moi['m01']/moi['m00'])))

        return bounds, com

    #shear
    elif testFlag == 2:
        centPts = []
        #define boundaries of RGB (in BGR order)
        boundaries = ([15,30,95], [90, 70, 255]) #red
        image = cv2.resize(image, None, fy = 1/6, fx=1/6, interpolation=cv2.INTER_CUBIC)
        imgW, imgH, channels = image.shape
        roiW = int(np.asarray(imgW) * 0.2)
        roiH = int(np.asarray(imgH) * 0.5)
        image = image[0:imgH, roiW:imgW] 

        lower = np.array(boundaries[0], dtype="uint8")
        upper = np.array(boundaries[1], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)
        edged = cv2.dilate(edged, None, iterations = 1)
        edged = cv2.erode(edged, None, iterations = 1)

        conts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conts = conts[1]
        (conts, _) = cnts.sort_contours(conts)

        for c in conts:
            if cv2.contourArea(c) < 100:
                continue 

            x1,y1,w1,h1 = cv2.boundingRect(c)
            midpt = ((x1+w1/2), (y1+h1))
            centPts.append(midpt)
            
        return centPts

    #bending
    else:
        linePts = []
        boundaries = ([15,30,95], [90, 70, 255]) #red
        image = cv2.resize(image, None, fy = 1/6, fx=1/6, interpolation=cv2.INTER_CUBIC)
        lower = np.array(boundaries[0], dtype="uint8")
        upper = np.array(boundaries[1], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 50, 150)
        edged = cv2.dilate(edged, None, iterations = 1)
        edged = cv2.erode(edged, None, iterations = 1)

        conts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conts = conts[1]
        (conts, _) = cnts.sort_contours(conts)

        for c in conts:
            if cv2.contourArea(c) < 100:
                continue 
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype='int')
            cv2.drawContours(image, [box.astype('int')], -1, (0, 255, 0), 2)

            (tl, tr, br, bl) = box
            (lx, ly) = ((tl[0] + tr[0])*0.5, (tl[1] + tr[1])*0.5)
            (rx, ry) = ((bl[0] + br[0])*0.5, (bl[1] + br[1])*0.5)
            subline = [(lx, ly), (rx, ry)] 
            cv2.line(image, (int(lx),int(ly)), (int(rx),int(ry)), (0, 255, 255), 2)
            linePts.append(subline)
            
        return linePts