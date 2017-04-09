import os
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours as cnts
import numpy as np
import imutils
import cv2
import ipdb

def midpt(ptA, ptB):
	return((ptA[0] + ptB[0])*0.5, (ptA[1] + ptB[1])*0.5)

def sizecalib(basepath, currentimage, flag):
	realWidth = 10 #actual size of white swatch in mm

	path = os.path.join(basepath, currentimage)
	image = cv2.imread(path)
	imgW, imgH, channels = image.shape
	if flag == 1:
		roiW = int(np.asarray(imgW) * 0.75)
		roiH = int(np.asarray(imgH) * 0.5)
		roiImage = image[roiH:imgH, 0:roiW]
		whiteBound = ([200, 200, 200], [255, 255, 255])
		areaThresh = 1000
	elif flag == 2: 
		image = cv2.resize(image, None, fy = 1/6, fx=1/6, interpolation=cv2.INTER_CUBIC)
		roiW = int(np.asarray(imgW)*0.50)
		roiH = int(np.asarray(imgH)*0.30)
		roiImage = image[0:roiH, 0:roiW]
		whiteBound = ([200, 200, 200], [255, 255, 255])
		areaThresh = 1000
	else:
		roiW = int(np.asarray(imgW)*0.75)
		roiH = int(np.asarray(imgH)*0.30)
		roiImage = image[roiH:imgH, 0:roiW]
		whiteBound = ([145, 145, 145], [255, 255, 255]) #shadowing is a problem
		areaThresh = 1000

	lower = np.array(whiteBound[0], dtype = 'uint8')
	upper = np.array(whiteBound[1], dtype = 'uint8')

	mask = cv2.inRange(roiImage, lower, upper)
	output = cv2.bitwise_and(roiImage, roiImage, mask = mask )
	gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	edged = cv2.Canny(gray, 50, 150)
	edged = cv2.dilate(edged, None, iterations = 1)
	edged = cv2.erode(edged, None, iterations = 1)

	contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if imutils.is_cv2() else contours[1]
	(contours, _) = cnts.sort_contours(contours)

	assert len(contours) != 0
	for c in contours:
		if cv2.contourArea(c) < areaThresh:
			continue

		orig = roiImage.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype='int')
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype('int')], -1, (0, 255, 0), 2)

		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

		(tl, tr, br, bl) = box
		(topX, topY) = midpt(tl, tr)
		(botX, botY) = midpt(bl, br)
		(leftX, leftY) = midpt(tl, bl)
		(rightX, rightY) = midpt(tr, br)

		distA = dist.euclidean((topX, topY), (botX, botY))
		distB = dist.euclidean((leftX, leftY), (rightX, rightY))

		if flag == 2:
			pxCalib = distA/realWidth
		else:	
			pxCalib = distB/realWidth

	return(pxCalib)