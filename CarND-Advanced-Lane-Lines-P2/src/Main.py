import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


import Pipeline


pLine = Pipeline.Pipeline()

testImage = cv2.imread('../test_images/test6.jpg')
testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
outImg = pLine.imagePipeline(testImage)

plt.imshow(outImg)
plt.show()