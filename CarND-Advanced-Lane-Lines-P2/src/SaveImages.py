import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

import ImageProcessing
import LineDetection



imgProc = ImageProcessing.ImageProcessing()
lineDet = LineDetection.LineDetection()

testImage = cv2.imread('../test_images/straight_lines1.jpg')
#testImage = cv2.imread('../test_images/test1.jpg')
testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
#testImage = cv2.imread('../camera_cal/calibration1.jpg')
undist = imgProc.undistortImage(testImage)

gradx = imgProc.absSobelThresh(undist, 'x', (20,100))
grady = imgProc.absSobelThresh(undist, 'y', (20,100))
mag_binary = imgProc.magThreshold(undist)
dir_binary = imgProc.dirThreshold(undist)
sChannelBinary = imgProc.sChannelThreshold(undist)
combinedBinaryImg = imgProc.makeBinaryImage(gradx, grady, mag_binary, dir_binary, sChannelBinary)

f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14, 10))
ax1.imshow(gradx, cmap = 'gray')
ax1.set_title('gradx')
ax2.imshow(grady, cmap = 'gray')
ax2.set_title('grady')
ax3.imshow(mag_binary, cmap = 'gray')
ax3.set_title('mag_binary')
ax4.imshow(dir_binary, cmap = 'gray')
ax4.set_title('dir_binary')
ax5.imshow(sChannelBinary, cmap = 'gray')
ax5.set_title('sChannelBinary')
ax6.imshow(combinedBinaryImg, cmap = 'gray')
ax6.set_title('combinedBinaryImg')

maskedImg = imgProc.getRegionOfInterest(combinedBinaryImg)


undistImagePerspectiveTransformed, M1, invM1 = imgProc.perspectiveTransform(undist)

imgSize = maskedImg.shape
offset = 80
yLim = 450
#srcPoints = np.float32([[160, imgSize[0]],[imgSize[1]//2 - offset, yLim],[imgSize[1]//2 + offset, yLim],[imgSize[1]-160, imgSize[0]]])
srcPoints = np.float32([[249, 690],[579, 460],[704, 460],[1058, 690]])
#srcPoints = np.float32([[263, 687],[618, 432],[661, 432],[1060, 687]])
maskedBinaryPerspectiveTransform, M2, invM2 = imgProc.perspectiveTransform(maskedImg, srcPoints)


f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14, 10))
ax1.imshow(testImage)
ax1.set_title('Original Image')
ax2.imshow(undist)
ax2.set_title('Undistorted Image')
ax3.imshow(combinedBinaryImg, cmap = 'gray')
ax3.set_title('Combined binary Image')
ax4.imshow(maskedImg, cmap = 'gray')
ax4.set_title('Masked binary Image')
ax4.plot(srcPoints[0][0], srcPoints[0][1], color = 'red', marker='o', markersize=3)
ax4.plot(srcPoints[1][0], srcPoints[1][1], color = 'green', marker='o', markersize=3)
ax4.plot(srcPoints[2][0], srcPoints[2][1], color = 'green', marker='o', markersize=3)
ax4.plot(srcPoints[3][0], srcPoints[3][1], color = 'red', marker='o', markersize=3)
ax5.imshow(undistImagePerspectiveTransformed)
ax5.set_title('Undistorted image perspective transform')
ax6.imshow(maskedBinaryPerspectiveTransform, cmap = 'gray')
ax6.set_title('Masked binary image perspective transform')


# leftx, lefty, rightx, righty, oImg, histogram = lineDet.findLanePixels(maskedBinaryPerspectiveTransform)
ploty, leftFitx, rightFitx, leftFitCoeffs, rightFitCoeffs, outImg = lineDet.fitPolynomial(maskedBinaryPerspectiveTransform)
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14, 10))
ax1.plot( np.sum(maskedBinaryPerspectiveTransform[maskedBinaryPerspectiveTransform.shape[0]//2:,:], axis=0))
ax1.set_title('Histogram') 

ax2.imshow(outImg)
ax2.plot(leftFitx, ploty)
ax2.plot(rightFitx, ploty)


#curvature
leftCurveRad, rightCurveRad = lineDet.measureCurvatureMeters(ploty, leftFitCoeffs, rightFitCoeffs)
ax2.set_title('Polynomial fit (Curvature: ' + str(np.round( (leftCurveRad + rightCurveRad) / 2) ) + ' m)') 

plt.show()