# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
- Make a pipeline that finds lane lines on the road
- Test the pipeline on images
- Test the pipeline on videos
<img src="./test_images_output/solidYellowCurve2.jpg " width="400">


[//]: # (Image References)



---

### Reflection

### 1. Pipeline desription

My pipeline consisted of 6 steps:
1. Convert the image to gray scale
<img src="./pipeline_images/grayscale.jpg " width="400">
2. Apply gaussian blur to smooth the image and remove noize
<img src="./pipeline_images/gaussianBlurImage.jpg " width="400">
3. Detect edges using Canny
<img src="./pipeline_images/edges.jpg " width="400">
4. Take in account only the region of interest, which is determined by a polygon with 4 vertices
<img src="./pipeline_images/maskedImage.jpg " width="400">
5. Apply Hough transformation to detect lines, average this lines and extrapolate them for whole region of interest
<img src="./pipeline_images/houghImage.jpg " width="400">
6. Merge this detected lines with the original image
<img src="./test_images_output/solidWhiteRight.jpg " width="400">

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by following steps:
1. Filter lines by their slope to avoid wrong detected lines and improve accuracy
2. Separate all detected lines by their slope: lines with negative slope are on the left side and lines with positive slope are on the right side
3. Calculate slope and intercept for each line  (line equation: y = mx + b, where m - slope, b - instersect)
4. Find standart deviation of m and b and filter out all lines, which deviate more than 68% (1 sigma) from the mean value to improve the accuracy
5. Extrapolate left and right lines for whole region of interst using the average line slope and intersect from the previous step





### 2. Shortcomings

* This algorithm was adjusted for the test videos. Processing of videos with different conditions (for example by rain or in the night or by higher traffic) could be not accurate enough with this pipeline
* If the road lines would have higher curvature, the algorithm may produce not correct results


### 3. Possible improvements

* Better tuning of parameters (Hough transformation parameters, Canny edge detection parameters)
* Apply more filtering algorithms to improve accuracy
* Improve the pipeline for lane lines with higher curvature
