import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

import ImageProcessing

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


# objpoints = [] # 3d points in real world space
# imgpoints = [] # 2d points in image plane.

# pickleIn = open("points.pickle", "rb")
# pickleData = pickle.load(pickleIn)
# objpoints = pickleData[0]
# imgpoints = pickleData[1]
# pickleIn.close()

# testImage = cv2.imread('../camera_cal/calibration1.jpg')
# grayTestImge = cv2.cvtColor(testImage,cv2.COLOR_BGR2GRAY)
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayTestImge.shape[::-1], None, None)
# dst = cv2.undistort(testImage, mtx, dist, None, mtx)
# #cv2.imshow('undistoted img',dst)
# #cv2.waitKey(0)
# plt.imshow(dst)
# plt.show()

imgProc = ImageProcessing.ImageProcessing()
testImage = cv2.imread('../test_images/straight_lines1.jpg')
#testImage = cv2.imread('../camera_cal/calibration1.jpg')
dst = imgProc.undistortImage(testImage)
plt.subplot(121)
plt.imshow(testImage)
plt.subplot(122)
plt.imshow(dst)
plt.show()