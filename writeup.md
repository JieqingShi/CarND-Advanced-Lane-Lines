## Project Writeup
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[test2]: ./test_images/test2.jpg "Test2"
[test2_warped]: ./output_images/test2.jpg_warped.jpg "Test2Warped"

---

## Writeup / README

### Disclaimer:
I started working on this project on the jupyter VM provided by udacity. However I ran into the problem that the binary images
used during the color thresholding part were not visualized correctly as it always outputted black images.
Therefore I have switched to working locally on an Anaconda environment. These are the packages and their respective versions
used:

| package       | version   | 
|:-------------:|:-------------:| 
| *numpy*      | 1.17.2      | 
| *matplotlib*      |   2.2.2    |
| *opencv-python*     |  4.1.0.25    |
| *moviepy*      | 0.2.3.2        |

### Camera Calibration

The code for this is included in the ``calib_camera.py`` module and in the first section of the `P2-Advanced-Lane-Lines.ipynb` notebook. The code is largely adopted from
 https://github.com/udacity/CarND-Camera-Calibration. Basically I'm going through all images in the ``camera_cal`` folder, convert
 them go grayscale and then finding corners in the chessboard pattern by calling the `cv2.findChessboardCorners()` function.
 If corners are found, the object points and corners are appended to the `objpoints` and `imgpoints` lists, respectively. 
I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix `mtx` and distortion coefficients `dist` 
using the `cv2.calibrateCamera()` function. These are saved as a pickle object to be used in the later steps of the pipeline.


### Pipeline (single images)

#### 1. Undistort and perspective transform
In the lectures the first steps of the pipeline consist of undistort - color threshold - perspective transform. 
However, since the order of operations between color thresholding and perspective transform does not matter for the final result, I have decided to
consolidate the undistortion and perspective transformation parts into one step. 

The code for this is found in in the `perspective_transform.py` module and in the second section of the notebook.
After loading in an image, camera calibration matrix and distortion coefficients the image is first undistorted using
`cv2.undistort()`. Then the undistorted image is converted to gray-scale and is warped into a birds-eye perspective
using `cv2.getPerspectiveTransform()`.

This is the result of this step:

_Original image_
![original image][test2]

_Warped image_
![!warped image][test2_warped]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
