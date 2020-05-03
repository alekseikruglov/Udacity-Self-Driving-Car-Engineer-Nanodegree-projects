import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

import ImageProcessing
import LineDetection


class Pipeline:

    def __init__(self):
        self.imgProc = ImageProcessing.ImageProcessing()
        self.lineDet = LineDetection.LineDetection()


    def imagePipeline(self, img):

        #undistort image
        undist = self.imgProc.undistortImage(img)

        #compute gradients
        gradx = self.imgProc.absSobelThresh(undist, 'x', (20,100))
        grady = self.imgProc.absSobelThresh(undist, 'y', (20,100))

        #create binary image for threshold magnitude
        mag_binary = self.imgProc.magThreshold(undist)
        #create binary image for threshold direction
        dir_binary = self.imgProc.dirThreshold(undist)
        #create binary image for S-Channel (HLS transform)
        sChannelBinary = self.imgProc.sChannelThreshold(undist)
        #combine all binry imges together
        combinedBinaryImg = self.imgProc.makeBinaryImage(gradx, grady, mag_binary, dir_binary, sChannelBinary)

        #apply mask to image to separte the are of interest
        maskedImg = self.imgProc.getRegionOfInterest(combinedBinaryImg)

        #transform perspective to "bird-eye" view
        maskedBinaryPerspectiveTransform, M, invM = self.imgProc.perspectiveTransform(maskedImg,np.float32([[249, 690],[579, 460],[704, 460],[1058, 690]]))

        #detect left nd right lines 
        #fit 2-nd order polynom for left aand right line
        #put the polynom lines on the imge 
        ploty, leftFitx, rightFitx, leftFitCoeffs, rightFitCoeffs, outImg = self.lineDet.fitPolynomial(maskedBinaryPerspectiveTransform)


        #Calculate curvature
        leftCurveRad, rightCurveRad = self.lineDet.measureCurvatureMeters(ploty, leftFitCoeffs, rightFitCoeffs)
        averageCurvature = np.round( (leftCurveRad + rightCurveRad) / 2) #calculate average curvature value for left an dright lines

        #draw the lane on the original image
        resultImg = self.imgProc.drawLineOnOriginalImage(maskedBinaryPerspectiveTransform, undist, ploty, leftFitx, rightFitx, invM)
        
        #put calculated curvature on the image
        cv2.putText(resultImg, 'Curvature: ' + str(averageCurvature) + ' m', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        #calculte car deviation fromlane center
        centerDev = self.lineDet.calculteCarDeviationFromLaneCenter(resultImg, leftFitx, rightFitx)

        #put calculated center deviation on the image
        cv2.putText(resultImg, 'Deviation from center: ' + str(centerDev) + 'm', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        return resultImg

