import numpy as np
import cv2

hls_dict = dict(H=0, L=1, S=2)
rgb_dict = dict(R=0, G=1, B=2)
luv_dict = dict(l=0, u=1, v=2)


def sobel_filter(img, orient="x", sobel_kernel=3):
    """ 
    Helper function for applying Sobel filter 
    
    Returns:
        sobel: Sobel filtered image using specified kernel size
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient=="x":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient=="y":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError("orient has to be x or y!!!")
    return sobel


def abs_sobel_thresh(img, orient='x', thresh=(0, 255), sobel_kernel=3):
    """ 
    Apply Sobel Filter on image gradient
    
    Returns: 
        binary_output: binary image after applying sobel filter in x or y direction
    """
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
    """
    Apply Filter on gradient magnitude
    
    Returns:
        binary_output: binary image filtered on gradient magnitude
    """
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
    """
    Apply Filter on gradient direction
    
    Returns:
        binary_output: binary image filtered on gradient direction
    """
    
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

def hls_thresh(img, hls_channel="S", thresh=(0, 255)):
    """
    Apply Filter on image in HLS color space
    
    Returns:
        binary_output: binary image filtered on specified channel in HLS color space
    """
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Apply a threshold to channel
    channel_num = hls_dict[hls_channel]
    channel_dim = hls[:,:,channel_num]
    # Return a binary image of threshold result
    binary_output = np.zeros_like(channel_dim)
    binary_output[(channel_dim > thresh[0]) & (channel_dim <= thresh[1])] = 1

    return binary_output

def rgb_thresh(img, rgb_channel="R", thresh=(0, 255)):
    """
    Apply Filter on image in RGB color space
    
    Returns:
        binary_output: binary image filtered on specified channel in RGB color space
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channel_num = rgb_dict[rgb_channel]
    channel_dim = rgb[:,:,channel_num]
    binary_output = np.zeros_like(channel_dim)
    binary_output[(channel_dim > thresh[0]) & (channel_dim <= thresh[1])] = 1
    
    return binary_output

