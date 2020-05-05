import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# Define a class to receive the characteristics of each line detection
class Line():

    # track last n detected lines
    n = 10

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recentXFitted = list()
        # polynomial coefficients
        self.polyCoeffs = list()
        #radius of curvature of the line in some units
        self.radiusOfCurvature = list() 
        #distance in meters of vehicle center from the line
        self.lineBasePos = list()
        
    def setRecentXFitted(self, xFitted):
        self.recentXFitted.append(xFitted)
        #if list length > n -> remove first (oldest) element
        if len(self.recentXFitted) > self.n:
            self.recentXFitted.pop(0)

    def setPolyCoeffs(self, coeffs):
        self.polyCoeffs.append(coeffs)
        #if list length > n -> remove first (oldest) element
        if len(self.polyCoeffs) > self.n:
            self.polyCoeffs.pop(0)

    def setRadiusOfCurvature(self, radius):
        self.radiusOfCurvature.append(radius)
        #if list length > n -> remove first (oldest) element
        if len(self.radiusOfCurvature) > self.n:
            self.radiusOfCurvature.pop(0)      

    def setLineBasePos(self, pos):
        self.lineBasePos.append(pos)
        #if list length > n -> remove first (oldest) element
        if len(self.lineBasePos) > self.n:
            self.lineBasePos.pop(0)   

    def getAveragXFitted(self):
        result = np.zeros_like(self.recentXFitted[0])
        for arr in self.recentXFitted:
            result = result + arr
        return result / len(self.recentXFitted)

    def getAveragePolyCoeffs(self):
        result = np.zeros_like(self.polyCoeffs[0])
        for arr in self.polyCoeffs:
            result = result + arr
        return result / len(self.polyCoeffs)

    def getAverageCurvRadius(self):
        result = np.zeros_like(self.radiusOfCurvature[0])
        for arr in self.radiusOfCurvature:
            result = result + arr
        return result / len(self.radiusOfCurvature)

    def getAverageLineBasePos(self):
        result = np.zeros_like(self.lineBasePos[0])
        for arr in self.lineBasePos:
            result = result + arr
        return result / len(self.lineBasePos)
