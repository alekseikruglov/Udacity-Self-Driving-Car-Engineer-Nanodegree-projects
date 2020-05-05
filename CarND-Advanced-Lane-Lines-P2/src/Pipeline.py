import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
import time

import ImageProcessing
import LineDetection
import Line


class Pipeline:

    def __init__(self):
        self.imgProc = ImageProcessing.ImageProcessing()
        self.lineDet = LineDetection.LineDetection()

        #previous detected lines (for tracking)
        self.leftLine = Line.Line()
        self.rightLine = Line.Line()


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

        #apply mask to image to separte the area of interest
        maskedImg = self.imgProc.getRegionOfInterest(combinedBinaryImg)

        #transform perspective to "bird-eye" view
        maskedBinaryPerspectiveTransform, M, invM = self.imgProc.perspectiveTransform(maskedImg,np.float32([[249, 690],[579, 460],[704, 460],[1058, 690]]))

        #detect left nd right lines 
        #fit 2-nd order polynom for left aand right line
        #put the polynom lines on the image 
        ploty, leftFitx, rightFitx, leftFitCoeffs, rightFitCoeffs, outImg = self.lineDet.fitPolynomial(maskedBinaryPerspectiveTransform, self.leftLine, self.rightLine)

        #check if the detected lines are real lane lines
        if self.lineDet.sanityCheck(leftFitx, rightFitx) or self.leftLine.detected == False:
            #add data to Line-objects
            self.leftLine.detected = True
            self.rightLine.detected = True

            self.leftLine.setRecentXFitted(leftFitx)
            self.rightLine.setRecentXFitted(rightFitx)

            self.leftLine.setPolyCoeffs(leftFitCoeffs)
            self.rightLine.setPolyCoeffs(rightFitCoeffs)

            #Calculate curvature
            leftCurveRad, rightCurveRad = self.lineDet.measureCurvatureMeters(ploty, self.leftLine.getAveragePolyCoeffs(), self.rightLine.getAveragePolyCoeffs())
            #averageCurvature = np.round( (leftCurveRad + rightCurveRad) / 2) #calculate average curvature value for left an dright lines

            #calculte car deviation from lane center
            centerDev = self.lineDet.calculteCarDeviationFromLaneCenter(outImg, self.leftLine.getAveragXFitted(), self.rightLine.getAveragXFitted())

            self.leftLine.setRadiusOfCurvature(leftCurveRad)
            self.rightLine.setRadiusOfCurvature(rightCurveRad)

            self.leftLine.setLineBasePos(centerDev)
            self.rightLine.setLineBasePos(centerDev)
        
        
        #draw the lane on the original image
        resultImg = self.imgProc.drawLineOnOriginalImage(maskedBinaryPerspectiveTransform, undist, ploty, self.leftLine.getAveragXFitted(), self.rightLine.getAveragXFitted(), invM)
        
        #put calculated curvature on the image
        averageCurvature = np.round( (self.leftLine.getAverageCurvRadius() + self.rightLine.getAverageCurvRadius() ) / 2) #calculate average curvature value for left and right lines
        cv2.putText(resultImg, 'Curvature: ' + str(averageCurvature) + ' m', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        
        #put calculated center deviation on the image
        cv2.putText(resultImg, 'Deviation from center: ' + str(np.round(self.leftLine.getAverageLineBasePos(), 2)) + 'm', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        
        return resultImg



    def videoPipeline(self, srcVideoPath, outVideoPath, startTime = None, endTime = None):

        if (startTime == None) and (endTime == None):
            clip1 = VideoFileClip(srcVideoPath)
        else:
            clip1 = VideoFileClip(srcVideoPath).subclip(startTime, endTime)

        project_clip = clip1.fl_image(self.imagePipeline) 
        project_clip.write_videofile(outVideoPath, audio=False)

