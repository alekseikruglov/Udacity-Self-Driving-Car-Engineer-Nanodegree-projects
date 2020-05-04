import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from IPython.display import HTML


import Pipeline


pLine = Pipeline.Pipeline()

# testImage = cv2.imread('../test_images/test1.jpg')
# testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
# outImg = pLine.imagePipeline(testImage)

# plt.imshow(outImg)
# plt.show()


# project_output = '../output_videos/project_video.mp4'
# clip1 = VideoFileClip("../project_video.mp4").subclip(0,3)

# project_clip = clip1.fl_image(pLine.imagePipeline) #NOTE: this function expects color images!!
# project_clip.write_videofile(project_output, audio=False)


pLine.videoPipeline("../project_video.mp4", '../output_videos/project_video.mp4')

