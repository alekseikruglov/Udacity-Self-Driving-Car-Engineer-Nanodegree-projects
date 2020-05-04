#includes methods for:
#camera calibration
#image undistortion
#perspective transform
#color transform
#gradient
#creating of binary image

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class ImageProcessing:

    def __init__(self):
        self.objpoints, self.imgpoints, self.mtx, self.dist = self.calibrateCamera()

    def calibrateCamera(self):
        # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # objp = np.zeros((6*9,3), np.float32)
        # objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # # Arrays to store object points and image points from all the images.
        # objpoints = [] # 3d points in real world space
        # imgpoints = [] # 2d points in image plane.

        # # Make a list of calibration images
        # images = glob.glob('../camera_cal/calibration*.jpg')

        # # Step through the list and search for chessboard corners
        # for fname in images:
        #     img = cv2.imread(fname)
        #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #     # Find the chessboard corners
        #     ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        #     # If found, add object points, image points
        #     if ret == True:
        #         objpoints.append(objp)
        #         imgpoints.append(corners)

        #         # Draw and display the corners
        #         '''
        #         img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #         cv2.imshow('img',img)
        #         cv2.waitKey(500)
        #         '''
                

        # cv2.destroyAllWindows()

        # pickleOut = open("points.pickle", "wb")
        # pickle.dump([objpoints, imgpoints], pickleOut)
        # pickleOut.close()


        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        pickleIn = open("points.pickle", "rb")
        pickleData = pickle.load(pickleIn)
        objpoints = pickleData[0]
        imgpoints = pickleData[1]
        pickleIn.close()

        img = cv2.imread('../camera_cal/calibration1.jpg')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return objpoints, imgpoints, mtx, dist


    def undistortImage(self, img):

        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        return dst

    def getRegionOfInterest(self, img):
        #mask image to separate unnecessary object for lane line detection
        
        #image dimensions
        imageShape = img.shape
        x = imageShape[1]
        y = imageShape[0]

        #create mask as polygon with vertices
        vertices = np.array([[(0,y),\
                                (0.45*x, 0.55*y),\
                                (0.55*x, 0.55*y),\
                                (x,y)]], dtype=np.int32)

        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (1,) * channel_count
        else:
            ignore_mask_color = 1
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def absSobelThresh(self, img, orient='x', thresh=(20, 100)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    def magThreshold(self, img, sobel_kernel=3, mag_thresh=(30, 100)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dirThreshold(self, img, sobel_kernel=3, dir_thresh=(0.7, 1.3)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def sChannelThreshold(self, img, thresh = (90,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output


    def makeBinaryImage(self, gradx, grady, magBinary, dirBinary, colorBinary):
        # combines  thresholds and color thresholds
        combined = np.zeros_like(dirBinary)
        combined[((gradx == 1) & (grady == 1)) | ((magBinary == 1) & (dirBinary == 1)) | (colorBinary == 1)] = 1

        return combined

    def perspectiveTransform(self, img, srcPoints = np.float32([[250, 690], [580,460], [705,460], [1060,690]]), offset = 0):

        #define destination points on transformed image
        xDst = img.shape[1]
        yDst = img.shape[0]
        dstPoints = np.float32([[srcPoints[0][0], yDst],\
                                [srcPoints[0][0], offset],\
                                [srcPoints[3][0], offset],\
                                [srcPoints[3][0], yDst]])

        #draw lines on image
        # lines = [[[srcPoints[i][0],
        #         srcPoints[i][1],
        #         srcPoints[(i+1) % len(srcPoints)][0],
        #         srcPoints[(i+1) % len(srcPoints)][1]]] for i in range(len(srcPoints))]

        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        invM = cv2.getPerspectiveTransform(dstPoints, srcPoints)   #nneded to transform the detected lines bck to original image
        # Warp the image using OpenCV warpPerspective()
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size)

        # Return the resulting image and matrix
        return warped, M, invM
        

    def drawLineOnOriginalImage(self, warped, undist, ploty, leftFitX, rightFitX, Minv):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([leftFitX, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightFitX, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        #plt.imshow(result)
        return result