import cv2
import numpy as np

def streetview2birdview(img, mtx, dist, src, dst, img_size):
    """ 
    Transforms image from regular view to bird-eye view 
    
    Returns: 
        warped: the bird-eye image
        M: transformation matrix
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)  # undistort
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)  # convert to gray-scale
    
    M = cv2.getPerspectiveTransform(src, dst)  # get perspective transform matrix
    warped = cv2.warpPerspective(undist, M, img_size)  # warp image
    return warped, M


def birdview2streetview(out_img, orig_img, combined_binary, dst, src, left_fitx, right_fitx, ploty):
    """ 
    Transforms image from bird-eye perspective to street view with lane lines highlighted
    
    Returns: 
        img_with_lane: original image with highlighted lane lines
    """
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_img.shape[1], orig_img.shape[0])) 

    img_with_lane = cv2.addWeighted(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), 1, newwarp, 0.3, 0)
    return img_with_lane


def show_crv_and_offset(img_with_lane, left_curverad, right_curverad, offset):
    """ 
    Print curvature and offset onto image
    
    """
    img_with_lane = cv2.putText(img_with_lane, f'Left lane curvature: {np.round(left_curverad, 2)} m ', 
                    (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    img_with_lane = cv2.putText(img_with_lane, f'Right lane curvature: {np.round(right_curverad, 2)} m', 
                    (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    # Display car offset
    img_with_lane = cv2.putText(img_with_lane, f'Offset: {np.round(offset, 2)} m', 
                    (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    return img_with_lane
    