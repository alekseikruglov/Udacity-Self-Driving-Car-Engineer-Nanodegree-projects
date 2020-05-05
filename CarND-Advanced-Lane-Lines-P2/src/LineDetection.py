# takes processed binry image (with perspective transform) and applies methods to detect left and right lines

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2



class LineDetection:

    # Define conversions in x and y from pixels space to meters
    ymPerPix = 30/720 # meters per pixel in y dimension
    xmPerPix = 3.7/700 # meters per pixel in x dimension

    # def __init__(self):
        


    def findLanePixels(self, img):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            # (win_xleft_high,win_y_high),(0,255,0), 2) 
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),
            # (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    

    def fitPolynomial(self, img, leftLineObj, rightLineObj):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, outImg = self.findLanePixels(img)

        # Fit a second order polynomial to each using `np.polyfit`
        # polynom coefficients for left and reight fit polynoms
        leftFitCoeffs = np.polyfit(lefty, leftx, 2)
        rightFitCoeffs = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        try:
            leftFitx = leftFitCoeffs[0]*ploty**2 + leftFitCoeffs[1]*ploty + leftFitCoeffs[2]
            rightFitx = rightFitCoeffs[0]*ploty**2 + rightFitCoeffs[1]*ploty + rightFitCoeffs[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            leftFitx = 1*ploty**2 + 1*ploty
            rightFitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        outImg[lefty, leftx] = [255, 0, 0]
        outImg[righty, rightx] = [0, 0, 255]

        return ploty, leftFitx, rightFitx, leftFitCoeffs, rightFitCoeffs, outImg

    # def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
    #     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    #     left_fit = np.polyfit(lefty, leftx, 2)
    #     right_fit = np.polyfit(righty, rightx, 2)
    #     # Generate x and y values for plotting
    #     ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    #     ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    #     left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #     right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    #     return left_fitx, right_fitx, ploty

    # def search_around_poly(self, binary_warped, left_fit, right_fit):
    #     # HYPERPARAMETER
    #     # Choose the width of the margin around the previous polynomial to search
    #     # The quiz grader expects 100 here, but feel free to tune on your own!
    #     margin = 100

    #     # Grab activated pixels
    #     nonzero = binary_warped.nonzero()
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
        
    #     ### TO-DO: Set the area of search based on activated x-values ###
    #     ### within the +/- margin of our polynomial function ###
    #     ### Hint: consider the window areas for the similarly named variables ###
    #     ### in the previous quiz, but change the windows to our new search area ###
    #     left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    #                     left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    #                     left_fit[1]*nonzeroy + left_fit[2] + margin)))
    #     right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    #                     right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    #                     right_fit[1]*nonzeroy + right_fit[2] + margin)))
        
    #     # Again, extract left and right line pixel positions
    #     leftx = nonzerox[left_lane_inds]
    #     lefty = nonzeroy[left_lane_inds] 
    #     rightx = nonzerox[right_lane_inds]
    #     righty = nonzeroy[right_lane_inds]

    #     # Fit new polynomials
    #     left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
    #     ## Visualization ##
    #     # Create an image to draw on and an image to show the selection window
    #     out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #     window_img = np.zeros_like(out_img)
    #     # Color in left and right line pixels
    #     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    #     # Generate a polygon to illustrate the search window area
    #     # And recast the x and y points into usable format for cv2.fillPoly()
    #     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    #     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
    #                             ploty])))])
    #     left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    #     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
    #                             ploty])))])
    #     right_line_pts = np.hstack((right_line_window1, right_line_window2))

    #     # Draw the lane onto the warped blank image
    #     # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #     # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    #     result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
    #     # Plot the polynomial lines onto the image
    #     # plt.plot(left_fitx, ploty, color='yellow')
    #     # plt.plot(right_fitx, ploty, color='yellow')
    #     ## End visualization steps ##
        
    #     return result

    def sanityCheck(self, leftFitx, rightFitx, distTollerance = 0.8, prallelTollerance = 0.5):
        # checks if the detected lines are the lane lines
        distOk = False
        parallelOk = False

        #check average distance between lines
        averXDist = np.mean(abs(leftFitx - rightFitx))
        if (3.7-distTollerance <= averXDist*self.xmPerPix) and (averXDist*self.xmPerPix <= 3.7+distTollerance):
            distOk = True

        #check lines are parallel
        beginDifference = np.mean(leftFitx[0:10]) - np.mean(rightFitx[0:10])    #distance between lines on the beginning
        endDifference = np.mean(leftFitx[-10:]) - np.mean(rightFitx[-10:])  #distance between lines on the end of lines
        if abs(beginDifference - endDifference)*self.xmPerPix < prallelTollerance:
            parallelOk = True

        return distOk and parallelOk




    def measureCurvaturePixels(self, img):
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''

        # fit polynoms and get polynom coefficients
        ploty, leftFitx, rightFitx, leftFitCoeffs, rightFitCoeffs, outImg = self.fitPolynomial(img)
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        yEval = np.max(ploty)
        
        # Calculation of R_curve (radius of curvature)
        leftCurveRad = ((1 + (2*leftFitCoeffs[0]*yEval + leftFitCoeffs[1])**2)**1.5) / np.absolute(2*leftFitCoeffs[0])
        rightCurveRad = ((1 + (2*rightFitCoeffs[0]*yEval + rightFitCoeffs[1])**2)**1.5) / np.absolute(2*rightFitCoeffs[0])
        
        return leftCurveRad, rightCurveRad

    def measureCurvatureMeters(self, ploty, leftFitCoeffs, rightFitCoeffs):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''

        # fit polynoms and get polynom coefficients
        # ploty, leftFitx, rightFitx, leftFitCoeffs, rightFitCoeffs, outImg = self.fitPolynomial(img)
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        yEval = np.max(ploty)
        
        # Calculation of R_curve (radius of curvature)
        leftCurveRad = ((1 + (2*leftFitCoeffs[0]*yEval*self.ymPerPix + leftFitCoeffs[1])**2)**1.5) / np.absolute(2*leftFitCoeffs[0])
        rightCurveRad = ((1 + (2*rightFitCoeffs[0]*yEval*self.ymPerPix + rightFitCoeffs[1])**2)**1.5) / np.absolute(2*rightFitCoeffs[0])
        
        return leftCurveRad, rightCurveRad


    def calculteCarDeviationFromLaneCenter(self, img, leftFitX, rightFitX):
        
        #points of the lines closest to the caar
        xLaneLeft = leftFitX[0]
        xLaneRight = rightFitX[0]

        #image center = camera center
        imgCenterPos = img.shape[1]//2

        #deviation from image center
        dev = abs(imgCenterPos - (abs(xLaneLeft - xLaneRight)/2 + xLaneLeft))

        #convert to meters, round to 2 decimals
        dev = np.round(dev*self.xmPerPix, 2)
        
        return dev




