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
        self.objpoints, self.imgpoints = self.calibrateCamera()

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

        return objpoints, imgpoints


    def undistortImage(self, img):
        grayIamge = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, grayIamge.shape[::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)

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
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
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


    def makeBinaryImage(self, gradx, grady, mag_binary, dir_binary):
        # combines  thresholds an color thresholds
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined

        # # Convert to HLS color space
        # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # l_channel = hls[:,:,1]
        # s_channel = hls[:,:,2]
        # # Sobel x
        # sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
        # abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # # Threshold x gradient
        # sxbinary = np.zeros_like(scaled_sobel)
        # sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # # Threshold color channel
        # s_binary = np.zeros_like(s_channel)
        # s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # # Stack each channel
        # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # return color_binary
  

    # def perspectiveTransform(self, img, nx, ny):
    #     # remove distortion
    #     undist = self.undistortImage(img)





    #     # Convert undistorted image to grayscale
    #     gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    #     # Search for corners in the grayscaled image
    #     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    #     if ret == True:
    #         # If we found corners, draw them! (just for fun)
    #         cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    #         # Choose offset from image corners to plot detected corners
    #         # This should be chosen to present the result at the proper aspect ratio
    #         # My choice of 100 pixels is not exact, but close enough for our purpose here
    #         offset = 100 # offset for dst points
    #         # Grab the image shape
    #         img_size = (gray.shape[1], gray.shape[0])

    #         # For source points I'm grabbing the outer four detected corners
    #         src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    #         # For destination points, I'm arbitrarily choosing some points to be
    #         # a nice fit for displaying our warped result 
    #         # again, not exact, but close enough for our purposes
    #         dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
    #                                     [img_size[0]-offset, img_size[1]-offset], 
    #                                     [offset, img_size[1]-offset]])
    #         # Given src and dst points, calculate the perspective transform matrix
    #         M = cv2.getPerspectiveTransform(src, dst)
    #         # Warp the image using OpenCV warpPerspective()
    #         warped = cv2.warpPerspective(undist, M, img_size)

    #     # Return the resulting image and matrix
    #     return warped, M