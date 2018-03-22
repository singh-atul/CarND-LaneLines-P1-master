# CarND-LaneLines-P1-master
Lane Detection Project of Term1 of Udacity Self Driving Car Course
# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md 

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


Creating a Great Writeup
---
For this project, a great writeup should provide a detailed response to the "Reflection" section of the [project rubric](https://review.udacity.com/#!/rubrics/322/view). There are three parts to the reflection:

1. Describe the pipeline

Pipeline will take the images one at a time . Its name in the code is **process_image** . It is common for both the video(multiple images) and for test image

This pipeline consists of 7 steps:


>>	1 : 
	Note * Lanes are of only of white or yellow color. So First we have to fetch all the white and yellow pixel from the image and leave other pixel as black.*
	
	'''python'''
	
def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)	
	
	RGB image is passed to **select_white_yellow** and the output is the RGB image that contains all the yellow and white pixel of the RGB image while rest all other in black

>> 2:
	Convert the RGB image from the select_white_yellow function into gray scale function using **cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)** 

>> 3:
	Smoothen the grayscale image by using Gaussian blur function in the HELPER FUNCTION BLOCK Kepping the kernel value to 15 
>> 4:
	Perform canny edge detection by using it in the HELPER FUNCTION BLOCK to get the boundary of the object in the image

>> 5:
	Determine the vertices so as to perform region masking.And then call the ROI function in the helper function block.The resultant image will contain the lanes
	which may be either a straight line or a broken lane
>> 6: Perform the hough line transform to get the lines in image . Set the parameters for the hough transform accordingly
	 rho=1, 
	 theta=np.pi/180
	 threshold=20
	 min_line_Gap=20
	 max_Line_Gap=300
	 
	 `image` should be the output of a Canny transform.
    
	lines        = hough_lines(regions, 1, np.pi/180, 20, 20, 300)
	
    Returns hough lines (not the image with lines)
	
	lines contains a list of lines detected.  With the above parameters, approximately 5-15 lines are detected for each image.

$$$  To draw line on the original image:- $$$

### Averaging and Extrapolating Lines

There are multiple lines detected for a lane line.  So to come up with an averaged line for that we will take the average of the lines.

Also, some lane lines are only partially recognized.  We should extrapolate the line to cover full lane line length.

We want two lane lines: one for the left and the other for the right.  The left lane should have a positive slope, and the right lane should have a negative slope.  Therefore, we'll collect positive slope lines and negative slope lines separately and take averages.

Note: in the image, `y` coordinate is reversed.  The higher `y` value is actually lower in the image.  Therefore, the slope is negative for the left lane, and the slope is positive for the right lane.


```python
def average_slope_intercept(lines):
    left_lines    = []
    left_weights  = [] 
    right_lines   = []
    right_weights = [] 
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)
```

Using the above `average_lines` function, we can calculate average slope and intercept for the left and right lanes of each image.  
 
$$$ To Draw Lanes $$$

We need to convert the slope and intercept into pixel points.

```python
def make_line_points(y1, y2, line):
    
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))
```
 
The ``draw_lines`` except a list of lines as the second parameter. In the given function make a separate image to draw lines and combine with the orignal later
image1 * α + image2 * β + λ
image1 and image2 must be the same shape.
	
 
 ### TEST IMAGES ###
 The images inputs are in test_image folder.  The image outputs are generated in test_image_output folder. Each image will be taken from the test image one at a time

```python``` 
import os
dir=os.listdir("test_images/")
directory='test_images_output'
if not os.path.exists(directory):
    os.makedirs(directory)
for i in dir:
    print(i)
    image = mpimg.imread('test_images/'+i)
    mpimg.imsave("test_images_output/"+i, process_image(image))

 #### TEST VIDEO ####
 

 The test_video_output folder will be created if not present .

 '''python'''
 
 clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(output, audio=False)

 The video inputs are in test_videos folder.  The video outputs are generated in test_videos_output folder. 
 
2. Identify any shortcomings

The project was successful in that the video images clearly show the lane lines are detected properly and lines are very smoothly handled.
It only detects straight and curved line. In case of steep changes like in hilly area we need to clearly distinguish between road and sky.



We encourage using images in your writeup to demonstrate how your pipeline works.  

All that said, please be concise!  We're not looking for you to write a book here: just a brief description.

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup. Here is a link to a [writeup template file](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md). 


The Project
---

## If you have already installed the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) you should be good to go!   If not, you should install the starter kit to get started on this project. ##

**Step 1:** Set up the [CarND Term1 Starter Kit](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/4f1870e0-3849-43e4-b670-12e6f2d4b7a7) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook

You will complete the project code in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  

**Step 3:** Complete the project and submit both the Ipython notebook and the project writeup

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

