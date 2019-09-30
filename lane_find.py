import numpy as np
import cv2


def hist(img):
    """
    Find histogram (aka sum of pixel values) of binary image from bottom half of image
    
    Returns:
        histogram: sum of pixel values of bottom half of pictures
    """
    bottom_half = img[img.shape[0]//2:,:] # 0:img.shape[0]//2 is the top half
    histogram = bottom_half.sum(axis=0) 
    
    return histogram


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    """
    Fit polynomial of second degree on image of size img_shape for left lane and right lane x and y coordinates
    
    Returns:
        left_fitx: x-values of fitted polynomial curve for left lane
        right_fitx: x-values of fitted polynomial curve for right lane
        ploty: y-values of fitted polynomial curve (same for left and right lane)
    """
    left_fit = np.polyfit(lefty, leftx, deg=2)
    right_fit = np.polyfit(righty, rightx, deg=2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty


def find_lanes_from_scratch(binary_warped, nwindows=9, margin=100, minpix=50, draw_lanes=True):
    """ 
    Detects lane pixels from scratch using sliding windows on the histogram peaks of the binary filtered image  
    
    Returns:
        left_fitx: x-values of fitted polynomial curve of detected left-lane
        right_fitx: x-values of fitted polynomial curve of detected right-lane
        left_fit: parameters of fitted polynomial of second order of left lane
        right_fit: parameters of fitted polynomial of second order of right lane
        ploty: y-values of fitted polynomial curve (same for left and right)
        out_img: image with marked pixels and sliding windows for left and right lane
        detected_flag: boolean flag, always True
    """
    # Find histogram peaks
    histogram = hist(binary_warped)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) # create output image to draw on (not necessary)
    midpoint = np.int(histogram.shape[0]//2)  # 640
    leftx_base = np.argmax(histogram[:midpoint])  # find index of left peak (indicates ll)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # find index of right peak (indicates rl)
    
    # Sliding windows
    window_height = np.int(binary_warped.shape[0]//nwindows)  # 80
    nonzero = binary_warped.nonzero()  # a tuple for x and y
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):  # index 0 to 8
        win_y_low = binary_warped.shape[0] - (window+1)*window_height  
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if draw_lanes:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) # bottom left to top right, in green, with thickness 2
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & 
                (nonzeroy >= win_y_low ) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & 
            (nonzeroy >= win_y_low ) & (nonzeroy < win_y_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)  # indices
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            # print(leftx_current)
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    # Find indices of left and right lane lines
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit polynomial of second degree
    left_fit = np.polyfit(lefty, leftx, deg=2)
    right_fit = np.polyfit(righty, rightx, deg=2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    detected_flag = True
    
    # Visualize
    if draw_lanes:
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        for index in range(out_img.shape[0]-1):
            cv2.line(out_img, (int(left_fitx[index]), int(ploty[index])), (int(left_fitx[index+1]), int(ploty[index+1])), (255,255,0), 3)
            cv2.line(out_img, (int(right_fitx[index]), int(ploty[index])), (int(right_fitx[index+1]), int(ploty[index+1])), (255,255,0), 3)
    return left_fitx, right_fitx, left_fit, right_fit, ploty, out_img, detected_flag


def find_lanes_from_prior(binary_warped, left_fit, right_fit, margin=100, draw_lanes=True):
    """
    Detects lane pixels by searching in margin around previous lane line position
    
    Returns:
        left_fitx: x-values of fitted polynomial curve of detected left-lane
        right_fitx: x-values of fitted polynomial curve of detected right-lane
        left_fit: parameters of fitted polynomial of second order of left lane (no modification done; passed directly from input)
        right_fit: parameters of fitted polynomial of second order of right lane (no modification done; passed directly from input)
        ploty: y-values of fitted polynomial curve (same for left and right)
        out_img: image with marked pixels and sliding windows for left and right lane
        detected_flag: boolean flag; is True if lane lines are found, False if not found
    
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
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
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    if leftx.size == 0 or rightx.size == 0:
        detected_flag = False
        return None, None, None, None, None, out_img, False  # super ugly
    else:
        detected_flag = True

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    if draw_lanes:
        window_img = np.zeros_like(out_img)
    
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        for index in range(out_img.shape[0]-1):
            cv2.line(out_img, (int(left_fitx[index]), int(ploty[index])), (int(left_fitx[index+1]), int(ploty[index+1])), (255,255,0), 3)
            cv2.line(out_img, (int(right_fitx[index]), int(ploty[index])), (int(right_fitx[index+1]), int(ploty[index+1])), (255,255,0), 3)
    
    return left_fitx, right_fitx, left_fit, right_fit, ploty, out_img, detected_flag



def measure_curvature(ploty, left_fitx, right_fitx):
    """
    Calculates the curvature of polynomial functions in meters.
    
    Returns: 
        left_curverad: curvature of left lane line
        right_curverad: curvature of right lane line
    
    """
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_fitx = left_fitx[::-1]
    right_fitx = right_fitx[::-1]
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    y_eval = np.max(ploty)
    
    
    left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(3/2)/(2*np.abs(left_fit_cr[0]))  
    right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(3/2)/(2*np.abs(right_fit_cr[0]))  
    
    return left_curverad, right_curverad


def measure_offset(left_fitx, right_fitx, midpoint=640):
    """
    Calculates offset from center of image from positions of left and right lane lines
    
    Returns:
        offset: offset from center
    """
    return (midpoint-(right_fitx[-1]+left_fitx[-1])/2)*3.7/700