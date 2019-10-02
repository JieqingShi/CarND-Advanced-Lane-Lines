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
[warped]: ./examples/warped.jpg "Test2Warped"
[color_thresh_example]: ./examples/color_thresh.jpg 
[combined_thresh_example]: ./examples/combined_thresh.jpg
[sliding_windows]: ./examples/sliding_windows.jpg
[lane_margin]: ./examples/lane_margin.jpg
[warped_back]: ./examples/warped_back.jpg
[project_video]: ./project_video_result.mp4

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


For this project I have provided several modules and in each module categorized by overall functionality, I have included the functions
needed for the lane detection pipeline in those modules.

- `calib_camera.py`: contains functions related to calibrating the camera
- `color_filtering.py`: contains various filters to create binary images (sobel filter, magnitude/gradient filter, color filters in HLS and RGB space)
- `lane_find.py`: contains the algorithms from the lectures (with some slight modifications) for detecting the lane lines, as well as functions for calculating the curvature and offset from center
- `perspective_transform.py`: contains functions for transforming the original images into birds-eye perspective (and back)

I have provided a notebook `P2-Advanced-Lane-Lines.ipynb` in which I am going through each step of the lane detection pipeline step-by-step, first on images, then on videos.
### Camera Calibration

The code for this part is included in the ``calib_camera.py`` module and in the first section of the `P2-Advanced-Lane-Lines.ipynb` notebook. The code is largely adopted from
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

All these steps are implemented in the `streetview2birdview()` function in the module.

The source and destination polygon points for warping the image are hardcoded and after playing around with some values, 
I have found these values to be quite reasonable:

| src       | dst   | 
|:-------------:|:-------------:| 
| 585, 460      | 350, 0        | 
| 203, 720      | 350, 720      |
| 1126, 720     | 950, 720      |
| 695, 460      | 950, 0        |

This is the result of this step:

_Original image_
![original image][test2]

_Undistored and Warped image_
![warped image][warped]

From the warped image one can see that both lane lines appear to be parallel.



#### 2. Color thresholding
I have implemented the various functions for filtering the image such as a gradient filter (Sobel) in x-direction, the gradient magnitude, 
 gradient direction, S channel from the HLS space and R channel from the RGB space filter. These are mainly adopted from the lecture and are included in the `color_filtering.py` module.

These result of the color filtering on the warped image is shown here: 

![color thresh example][color_thresh_example]

I have tried several combinations of gradient filtering and color filtering.
While experimenting with images I was initially under the assumption that a combined S channel filter from the HLS color space
and R channel filter from the RGB color space performed the best. 
To get the combined image, I generate an S channel filtered and an R channel filtered image, ``s_binary`` and ``r_binary``, respectively and combined  the image using 
```python
# Combine thresholds, e.g. using the S and R channel
combined_binary = np.zeros_like(s_binary)
combined_binary[((s_binary == 1) | (r_binary==1))] = 1

```

This is the combined S and R filter applied on the warped image from [warped]:
![combined thresh_example][combined_thresh_example]


(Although it looks like as if the combined filter is perfectly able to outline both lanes, during the experimentation with other images I have found the R channel filter to be too noisy and not robust against changing lighting environments.
I have corrected this for the video pipeline)

#### 4. Finding lane lines
As in the lecture, I have implemented two methods for finding lane lines (which can be found in the `find_lanes.py` module)

- by scratch: finding histogram peaks in the bottom half of the color thresholded and warped image, searching around lane pixels 
by iterating through sliding windows to track the curvature and then fit a polynomial of second order. This is adapted from the lecture 
and can be found in the `find_lanes_from_scratch()` function.

This is what it looks like when applying the method on the color-thresholded and warped image:
![sliding windows][sliding_windows]

The red and blue lines denoted pixels of the left and right lane, respectively. The yellow lines are the fitted polynomials around those pixels,
and the green rectangles are the sliding windows

- from prior: search around within a margin from polynomial lines obtained from previously detected lanes. This method is found in 
the `find_lanes_from_prior()` function.

This is the result when the function is applied on the example image, with the green border showing the margin around which the lane pixels are searched.
![lane margin][lane_margin]


#### 5. Calculating curvature radius and offset from lane center.

For the curvature, I am using the same method from the lecture and the same conversion rate from pixels to meters
of 30/720 for the y dimension and 3.7/700 in x dimension. I am calculating curvatures for left and right lane separately (although
one could simply consolidate both values by taking the mean). 

For the vehicle distance to the center of the lane, I am calculating the vehicle position by taking the mean of the first pixel of the left and right lane, respectively.
Then, assuming the camera is mounted on the center, I am subtracting that value from the midpoint of the image, upon which I obtain
the distance in pixel values. This value can then be simply converted to meters by multiplying it with the conversion factor 3.7/700.

The calculations are implemented in the `measure_curvature()` and `measure_offset()` functions in the `lane_find.py` module.

#### 6. Warping detected lanes back onto the road such that the lane area is identified clearly.

This is implemented in the `birdview2streetview()` function in the `perspective_transform.py` module, basically the inverse operation from the warping in step 1 of the pipeline.
First we calculate the inverse transformation matrix from the `dst` and `src` matrix defined for step 1 of the pipeline.
Using the inverse Matrix `Minv` we can warp back the detected lanes from birds-eye perspective back onto the original image using `cv2.warpPerspective()` and color the area between left and right lane.


This is what the final result looks like:

![warped back][warped_back]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
For the video pipeline I first started by simply taking all the above steps and wrapping them inside a function. 
This basically goes through the entire lane detection pipeline
on a frame-to-frame basis.

As for detecting the lines either from prior frames or from scratch, I found a way to test this by 
setting a small search margin (around 10 to 20 pixels) for the `find_lanes_from_prior()` function. I set up an if-else statement which 
calls the function for finding lanes from prior. However due to the small search margin, it is likely that no lane lines could be found from the prior search. Therefore it would call to the `find_lanes_from_scratch()` function
. 

I also realized that using the method for detecting lanes from prior does not necessarily need to detect a lane for each frame,
(e.g. in case of a bad frame with lots of shadows) while the method for detecting lanes from scratch always has to find lane lines.
This is why I included, as an additional output, a boolean flag into each of the two functions which indicates if a lane has been detected or not.
For the `find_lanes_from_scratch()` function the flag is always set to `True` while for the `find_lanes_from_prior()` function, the flag 
can be `False`.


Going through the simple video pipeline I found that it performed reasonably well on the project video (predominantly straight roads), but not on the challenge video (some small curves and different lighting).

The detected lanes in the challenge video were jumping from one frame to another and overall, the lanes were not detected reliably (e.g. lots of 
frames where the lanes were intersecting or where the lane lines were completely off)

This is when I realized that a) the R channel filter is not suited for images with weird lighting or shadows as it produces
noisily filtered images and b) that a mechanism needs to be implemented from preventing the lines from jumping around from frame to frame.

This is why I improved the pipeline by using a different combination of filters by using the gradient filter in x direction, the 
gradient magnitude and direction filter as well as the S-channel filter. They were combined as follows:

```python 
    grad_binary = abs_sobel_thresh(warped_img, orient='x', thresh=(20, 100), sobel_kernel=3)
    mag_binary = mag_thresh(warped_img, thresh=(30, 100), sobel_kernel=3)
    dir_binary = dir_thresh(warped_img, thresh=(0.7, 1.3), sobel_kernel=15)
    s_binary = hls_thresh(warped_img, hls_channel="S", thresh=(90, 255))
    combined_binary = np.zeros_like(dir_binary)
    combined_binary[(grad_binary == 1) | (s_binary == 1) | ( ((mag_binary == 1) & (dir_binary == 0)))] = 1
```
This combined filter appeared to be more robust to changing environments.

Subsequently I also introduced a `Line()` class to store some of the calculations inside the classes attributes.
I used a slightly different implementation than the one proposed in the lecture. 

For instance, I introduced a counter which increments if a lane has not been found in one frame.
If that counter exceeds a certain threshold, I would trigger a search for lane lines from scratch. Otherwise 
I am relying on finding the lane lines from prior frames.

Furthermore, I implemented a smoothing mechanism by introducing a frame buffer of 12 frames as a class attribute. For each frame that is processed,
I am storing the results of the fitted x values of the lane polynoms in this class attribute.
This list is updated in a FIFO schedule (the oldest one being replaced by the most recent one), thus this list always has a length of 12 (except for the first 11 frames of the video).
For the current frame, in order to calculate the lane polynoms I am simply taking the median of 
all the polynoms of the twelve most recent frames in this list. I chose to use the median 
instead of the mean in order to counter potential outliers.


Through this additional feature the detected lanes appear much smoother and less "jumpy".
Of course since this is 
a tunable parameter one can play around with the most optimal frame buffer value to achieve better results (longer frame
buffers make the lanes more smooth but also possibly more "sluggish", i.e. reacting slower to changes in curves)

Due to these changes my results on the challenge video have improved.


Here's a [link to my result of the project video](./project_video_result.mp4).
This is [a link to the challenge video](./challenge_video_result.mp4)

The code for the video pipeline can be found in the `process_video()` function in the notebook.

---

### Discussion

Despite the changes to my video pipeline the lane detection algorithm still does not perform well on the [harder challenge video](./harder_challenge_video_result.mp4). 
This might be because in the harder challenge videos the changes in lighting and shadows are much
more drastic while there are also much more prominent curves (sometimes extreme curves which disappear outside the frame).

In overall I can think of the following proposals to improve my implementation:

- separate the search of the left and right lane lines: currently both are done in the same function. 
However it could be the case that only one of the lane lines is found. In that case there is no need to trigger a re-search from scratch for both lane lines, but only for the one which has not been found.

- deal with intersecting lane lines: sometimes when the lines are detected incorrectly (left lane curves to the right, right lane curves to the left), the left lane and right lane will intersect. In this case one could add a sanity check for detecting cases where the minimum x-pixel of the right lane is smaller than the maximum x-pixel of the left lane (meaning intersection). When that happens one could trigger a search from scratch or use a smaller src/dst polygon for the warping

- calculate lane width: an additional check would be to calculate lane width (right lane x-pixel minus left lane x-pixel); if it's too wide (e.g. right lane detected outside the actual lane) one could start a search from scratch


- alternative approach to smoothing lanes across frames: in my current approach I am returning the left_fitx and right_fitx values from the already fit polynomes and average across those values; an alternative approach would be to return the leftx and rightx pixel values, average across those and then fit a polynom on the averaged pixels


- diverging curvatures: if the curvatures of left and right lane lines diverge too much (let's say a factor of 10), one can mirror the polynom of either the left or right lane to the other side


- fit more complex polynomial: under extreme curve situations (tight left curve followed directly by tight right curve), a polynomial of second degree does not properly represent the curvature anymore, in that case a more complex polynom might be more fitting


- better parameter tuning: there are many parameters which can be more carefully tuned (such as the size of the warping polynoms src and dst, the thresholds for the filters etc.). Or one could use a set of multiple parameters for multiple situations or even dynamic parameters.