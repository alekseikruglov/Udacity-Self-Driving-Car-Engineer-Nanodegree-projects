import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from IPython.display import HTML


import Pipeline


pLine = Pipeline.Pipeline()

# process image
# testImage = cv2.imread('../test_images/straight_lines1.jpg')
# testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
# outImg = pLine.imagePipeline(testImage)

# f = plt.figure()
# plt.imshow(outImg)
# f.savefig('../output_images/straight_lines1.png', dpi=f.dpi, bbox_inches='tight')

# plt.show()


#process video
pLine.videoPipeline("../project_video.mp4", '../output_videos/project_video.mp4')


