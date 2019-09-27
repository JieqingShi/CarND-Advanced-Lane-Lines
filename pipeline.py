"""
Module containing functions for the entire image transformation pipeline. This assumes that the camera has already been calibrated
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

# def undistort_image(img, mtx, dist):
#     dst = cv2.undistort(img, mtx, dist, None, mtx)
#     return dst
hls_dict = dict(H=0, L=1, S=2)

def perspective_transform(img, mtx, dist, src, dst, img_size):
    """ Transforms image to bird-eye view """
    undist = cv2.undistort(img, mtx, dist, None, mtx)  # undistort
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)  # convert to gray-scale
    
    M = cv2.getPerspectiveTransform(src, dst)  # get perspective transform matrix
    warped = cv2.warpPerspective(undist, M, img_size)  # warp image
    return warped, M


def sobel_filter(img, orient="x", sobel_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient=="x":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient=="y":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError("orient has to be x or y!!!")
    return sobel

# Can be implemented via property?
def abs_sobel_thresh(img, orient='x', thresh=(0, 255), sobel_kernel=3):
    """ Apply Sobel Filter in x or y direction """
    sobel = sobel_filter(img, orient=orient, sobel_kernel=sobel_kernel)
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create mask of ones
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1  # ToDo: use min max instead of index
    
    return binary_output


def mag_thresh(img, thresh=(0, 255), sobel_kernel=3):
    
    sobelx = sobel_filter(img, orient="x", sobel_kernel=sobel_kernel)
    sobely = sobel_filter(img, orient="y", sobel_kernel=sobel_kernel)
    # Calculate the magnitude
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
 
    return binary_output


def dir_thresh(img, thresh=(0, np.pi/2), sobel_kernel=3):
    
    sobelx = sobel_filter(img, orient="x", sobel_kernel=sobel_kernel)
    sobely = sobel_filter(img, orient="y", sobel_kernel=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(sobelx)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    
    return binary_output

def hls_thresh(img, hls_channel="S", thresh=(0, 255)):  # ToDo: can be generalized to any color space
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Apply a threshold to channel
    channel_num = hls_dict[hls_channel]
    channel_dim = hls[:,:,channel_num]
    # Return a binary image of threshold result
    binary_output = np.zeros_like(channel_dim)
    binary_output[(channel_dim > thresh[0]) & (channel_dim <= thresh[1])] = 1

    return binary_output

def hist(img):
    bottom_half = img[img.shape[0]//2:,:] # 0:img.shape[0]//2 is the top half
    histogram = bottom_half.sum(axis=0)  # just sum, no division!
    
    return histogram


def find_lane_pixels(binary_warped):
    print(binary_warped.shape)  # 720 x 1280
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)  # 640
    #print("midpoint: {}".format(midpoint))
    leftx_base = np.argmax(histogram[:midpoint])  # 327
    #print("leftx_base: {}".format(leftx_base))
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # 971
    #print("rightx_base: {}".format(rightx_base))

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9  # number of windows on vertical axis
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)  # 80
    # print("window_height: {}".format(window_height)) 
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()  # a tuple for x and y
    nonzeroy = np.array(nonzero[0])
    #print((nonzeroy.max(), nonzeroy.min()))  # (719, 0)
    #print("length nonzeroy {}".format(len(nonzeroy)))  # 89486 nonzero pixel elements found
    nonzerox = np.array(nonzero[1])
    #print((nonzerox.max(), nonzerox.min()))  # (1279, 0)
    #print("length nonzerox {}".format(len(nonzerox)))  # 89486
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):  # index 0 to 8
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height  # go up the picture starting from picture 640 in steps of 80 to 0
        # top edge of the window
        #print("win_y_low: {}".format(win_y_low))
        win_y_high = binary_warped.shape[0] - window*window_height  # 80 more than win_y_low, goes up the picture up to 80
        # bottom edge of the window
        # print("win_y_high: {}".format(win_y_high))
        ### TO-DO: Find the four below boundaries of the window ###
        
        # corresponds to four corners
        #win_xleft_low = 0  # Update this
        #win_xleft_high = 0  # Update this
        #win_xright_low = 0  # Update this
        #win_xright_high = 0  # Update this
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), 
        (win_xleft_high,win_y_high),(0,255,0), 2) # bottom left to top right, in green, with thickness 2
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        #good_left_inds = None
        #good_right_inds = None
        # all conditions return False, True of same length as nonzerox/y
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & 
            (nonzeroy >= win_y_low ) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & 
            (nonzeroy >= win_y_low ) & (nonzeroy < win_y_high)).nonzero()[0]
        # index 0 because we are only interested in the horizontal position
        # shows which element in nonzerox are nonzero
        # print(good_left_inds.min(), good_left_inds.max())  # why does it go up to 82000??
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)  # indices
        right_lane_inds.append(good_right_inds)
        #print(len(good_left_inds))
        #print(len(good_right_inds))
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        # nonzero operates on an index level! so get exact pixel value based on index of nonzero!
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            # print(leftx_current)
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            # print(rightx_current)
        #pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    #left_fit = None
    #right_fit = None

    left_fit = np.polyfit(leftx, lefty, deg=2)
    right_fit = np.polyfit(rightx, righty, deg=2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(leftx, lefty, deg=2)
    right_fit = np.polyfit(rightx, righty, deg=2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox >= left_fit[0]*nonzeroy**2 + 
        left_fit[1]*nonzeroy + left_fit[2] - margin) & 
        (nonzerox < left_fit[0]*nonzeroy**2 + 
        left_fit[1]*nonzeroy + left_fit[2] + margin)).nonzero()[0]
    right_lane_inds = ((nonzerox >= right_fit[0]*nonzeroy**2 + 
        right_fit[1]*nonzeroy + right_fit[2] - margin) & 
        (nonzerox < right_fit[0]*nonzeroy**2 + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)).nonzero()[0]
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result


def measure_curvature_pixels():
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit, right_fit = generate_data()
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    #left_curverad = 0  ## Implement the calculation of the left line here
    #right_curverad = 0  ## Implement the calculation of the right line here
    left_curverad = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2))**(3/2)/(2*np.abs(left_fit[0]))
    right_curverad = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2))**(3/2)/(2*np.abs(right_fit[0]))
    
    return left_curverad, right_curverad


def measure_curvature_real():
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(3/2)/(2*np.abs(left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(3/2)/(2*np.abs(right_fit_cr[0]))  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad
